// src/poly/inkling2.cpp - the layers of Inkling 2 Mini.  The reference for the math is llama.cpp's
// inkling2 graph (docs/models/reference/inkling2.cpp, and inkling2-loop.cpp for the looped core).
//
// A token runs 48 logical layers: the prelude [0, 6), the looped core [6, 18) three times over the same
// weights, then the epilogue [18, 24).  Each visit has its own attention cache (8 full-context, 40 windowed)
// and its own short-convolution history, and a pass hands the next one four rank-64 messages a core layer.
#include "inkling2.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 1e-20f;   // this model's expert_weights_norm_eps
constexpr float kMixEps = 1e-6f;         // mix() and control() norm with 1e-6, not the model's eps
constexpr int kCtrl = 4;                  // the looped core's controllers: q, k, v, ffn

int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size())
        throw std::runtime_error("inkling2_loop: a per-layer metadata array ends before block_count");
    return (int64_t) v.items[(size_t) il].u;
}

/// A per-layer value the model has for every layer, which must be the same one throughout: the engine
/// keeps one of it for all of them.
int64_t uniform(const MetaValue& v, int n_layer, const char* what) {
    const int64_t first = of_layer(v, 0);
    for (int il = 1; il < n_layer; ++il)
        if (of_layer(v, il) != first)
            throw std::runtime_error(std::string("unsupported inkling2_loop model: varying ") + what);
    return first;
}

const char* const kCtrlName[kCtrl] = {"q", "k", "v", "ffn"};

}  // namespace

void Inkling2::begin_pass() {
    // begin_pass() runs twice at load, before load_layer() fills the blocks, and once per call: the weights
    // stay, and only what a call leaves behind is reset.
    if (blocks_.size() != layers_.size()) {
        blocks_.assign(layers_.size(), Block{});
        loops_.assign((size_t) std::max(0, core_end_ - core_start_), Loop{});
    }
    n_bank_ = 0;
    n_prev_bank_ = 0;
}

// The conv history of a visit: sconv_len_ - 1 rows of K, then V, then the attention's update, then the FFN's.
float* Inkling2::past(int s) const {
    return state_ + (size_t) s * (size_t) (sconv_len_ - 1) * (size_t) past_channels();
}

int Inkling2::past_at(int which) const {
    const int k = sconv_len_ - 1, chan_kv = n_kv_ * head_dim_;
    return which == 0 ? 0 : which == 1 ? k * chan_kv : which == 2 ? 2 * k * chan_kv : k * (2 * chan_kv + n_embd_);
}

void Inkling2::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported inkling2_loop model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };
    need(opt_.layer_split.empty(), "--layer-split (the looped core spans the split)");
    // int8/int8k are Gemma 4's layout; inkling2_loop supports fp16 (default) and q8_0.
    need(opt_.kv_format != KvFmt::kInt8 && opt_.kv_format != KvFmt::kInt8K,
         "--kv int8 or int8k (use fp16 or q8_0 for inkling2_loop)");

    const MetaValue& hc = must("attention.head_count");
    const MetaValue& hkv = must("attention.head_count_kv");
    n_head_ = (int) uniform(hc, n_layer_, "attention.head_count");
    n_kv_ = (int) uniform(hkv, n_layer_, "attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    need(n_head_ > 0 && n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad attention.head_count(_kv)");
    need(shape_i("attention.value_length") == head_dim_, "keys and values have different widths");
    need(head_dim_ % 32 == 0 && head_dim_ <= 256, "bad head size");

    window_ = shape_i("attention.sliding_window");
    need(window_ > 0, "bad attention.sliding_window");
    d_rel_ = shape_i("attention.relative_length");
    rel_extent_ = shape_i("attention.relative_extent");
    qk_clip_ = shape_f("attention.qk_clip");
    rel_clip_ = shape_f("attention.relative_clip");
    log_scale_floor_ = shape_i("attention.log_scale_floor");
    log_scale_alpha_ = shape_f("attention.log_scale_alpha");
    sconv_len_ = shape_i("shortconv.l_cache");
    attn_res_block_ = shape_i("attn_res.block_size");
    final_softcap_ = shape_f("final_logit_softcapping");
    need(d_rel_ > 0 && rel_extent_ > 0, "bad attention.relative_length/extent");
    need(sconv_len_ >= 2 && sconv_len_ <= 64, "shortconv.l_cache is not in [2, 64]");
    need(attn_res_block_ > 0, "bad attn_res.block_size");
    need(log_scale_floor_ > 0 && log_scale_alpha_ >= 0.f, "bad attention.log_scale_floor/alpha");
    need(qk_clip_ > 0.f && rel_clip_ > 0.f, "bad attention.qk_clip/relative_clip");

    // the routed experts are activated by situ, not SwiGLU: the engine's moe_act() takes that path, and
    // cpu_experts()'s too when situ_ is set, so the SwiGLU limits are never read (INFINITY, not an empty vector)
    situ_ = true;
    situ_beta_ = shape_f("activation.situ_beta");
    situ_linear_beta_ = shape_f("activation.situ_linear_beta");
    need(situ_beta_ > 0.f && situ_linear_beta_ > 0.f, "bad activation.situ_beta/situ_linear_beta");
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;

    // the experts are the same in every layer, which the engine keeps one geometry of
    n_used_ = (int) uniform(must("expert_used_count"), n_layer_, "expert_used_count");
    n_ff_exp_ = (int) uniform(must("expert_feed_forward_length"), n_layer_, "expert_feed_forward_length");
    n_ff_shexp_ = n_ff_exp_;   // no shared expert: moe_shared() writes zeros

    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : true;   // llama.cpp's graph normalizes them, key or not
    const MetaValue* scale = find("expert_weights_scale");
    route_scale_ = scale ? (float) scale->num() : 0.f;

    // the embedding and the output are padded past the vocabulary, and the reference drops the pad
    const MetaValue& unpad = must("vocab_size_unpadded");
    n_vocab_unpadded_ = (int) unpad.u;
    need(n_vocab_unpadded_ > 0 && n_vocab_unpadded_ <= n_vocab_,
         "bad vocab_size_unpadded (above vocab_size)");

    // The release that ships this model counts its blocks logically: its block_count is the 48 applications a
    // token makes, and the file carries the 24 physical blocks, which its own tensors say.  A file that counts
    // the physical ones, as the plan's do, walks no steps here.
    while (n_layer_ > 1 && !tensor(n_layer_ - 1, "attn_k.weight")) --n_layer_;

    // situ bounds both sides of the activation, so no SwiGLU limit applies to either expert path
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;

    // this release renamed the loop's keys, so both spellings name the same numbers
    auto loop_shape = [&](const char* now, const char* was) -> int {
        if (find(now)) return shape_i(now);
        if (find(was)) return shape_i(was);
        throw std::runtime_error(std::string("unsupported inkling2 model: no ") + now + " (nor " + was + ")");
    };
    core_start_ = loop_shape("loop.core_start", "loop.start");
    core_end_ = loop_shape("loop.core_end", "loop.end");
    loop_count_ = loop_shape("num_loops", "loop.count");
    gate_rank_ = shape_i("loop.gate_rank");
    need(core_start_ > 0 && core_start_ < core_end_ && core_end_ <= n_layer_, "bad loop.core_start/loop.core_end");
    need(loop_count_ >= 2, "bad num_loops (a fixed loop needs two passes)");
    need(gate_rank_ > 0 && gate_rank_ <= 256, "bad loop.gate_rank");
    // llama.cpp's load_arch_hparams() refuses a core that is not a whole number of attn_res blocks
    need(core_start_ % attn_res_block_ == 0 && core_end_ % attn_res_block_ == 0,
         "the loop core is not a whole number of attn_res blocks");
    n_prelude_blocks_ = core_start_ / attn_res_block_;
    n_core_blocks_ = (core_end_ - core_start_) / attn_res_block_;
    n_epilogue_blocks_ = (n_layer_ - core_end_) / attn_res_block_;

    const MetaValue& swa = must("attention.sliding_window_pattern");
    swa_pattern_.resize((size_t) n_layer_);
    for (int il = 0; il < n_layer_; ++il) swa_pattern_[(size_t) il] = of_layer(swa, il) != 0 ? 1 : 0;

    // The unrolled schedule: the prelude [0, core_start_), the core [core_start_, core_end_) loop_count_
    // times, the epilogue [core_end_, n_layer_).  Inkling 2 Mini: 6 + 12*3 + 6 = 48 visits.
    steps_.clear();
    auto push = [&](int phys, int pass) {
        Step st;
        st.phys = phys;
        st.pass = pass;
        st.swa = swa_pattern_[(size_t) phys] != 0;
        st.local = st.swa ? (int) std::count_if(steps_.begin(), steps_.end(),
                                                [](const Step& o) { return o.swa; })
                          : -1;
        steps_.push_back(st);
    };
    for (int il = 0; il < core_start_; ++il) push(il, 0);
    for (int p = 0; p < loop_count_; ++p)
        for (int il = core_start_; il < core_end_; ++il) push(il, p);
    for (int il = core_end_; il < n_layer_; ++il) push(il, 0);

    // The depth attention's bank at its largest: a block start in the prelude, the core's in the first pass
    // and the epilogue's, and the layer's own input last.
    max_states_ = n_prelude_blocks_ + n_core_blocks_ + n_epilogue_blocks_ + 1;
    // a full-context layer's bias table reaches the relative extent, a window layer's its window
    max_keys_ = std::max(window_, rel_extent_) + 1;
    n_bank_ = 0;
    n_prev_bank_ = 0;

    std::fprintf(stderr,
                 "strata-poly: inkling2_loop: %d physical -> %zu logical layers (%zu full, %zu SWA=%d), conv %d, "
                 "core [%d, %d) x %d, depth %d states\n",
                 n_layer_, steps_.size(), steps_.size() - (size_t) std::count_if(steps_.begin(), steps_.end(),
                                                                                [](const Step& s) { return s.swa; }),
                 (size_t) std::count_if(steps_.begin(), steps_.end(), [](const Step& s) { return s.swa; }), window_,
                 sconv_len_, core_start_, core_end_, loop_count_, max_states_);
}

void Inkling2::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    auto have = [&](const char* n) { return tensor(il, n) != nullptr; };

    B.swa = swa_pattern_[(size_t) il] != 0;
    B.block_start = il % attn_res_block_ == 0;
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    // The attention's launches read the position when they are recorded (a full-context layer's log-scale and a
    // window layer's ring both do), so keep the recordings off until Phase 3 sizes them.
    L.fixed = false;

    const int64_t hq = (int64_t) n_head_ * head_dim_, hkv = (int64_t) n_kv_ * head_dim_;
    B.wq = mat("attn_q.weight", n_embd_, hq);
    B.wk = mat("attn_k.weight", n_embd_, hkv);
    B.wv = mat("attn_v.weight", n_embd_, hkv);
    B.wo = mat("attn_output.weight", hq, n_embd_);
    B.w_gate = mat("attn_gate.weight", n_embd_, hq);
    B.q_clip = f32("attn_q_clip.weight", head_dim_, n_head_);
    B.rel = mat("attn_rel.weight", n_embd_, (int64_t) d_rel_ * n_head_);
    B.rel_clip = f32("attn_rel_clip.weight", d_rel_, n_head_);
    B.sink_proj = mat("attn_sink_proj.weight", n_embd_, n_head_);
    B.k_conv = f32("attn_k_conv.weight", sconv_len_, hkv);
    B.v_conv = f32("attn_v_conv.weight", sconv_len_, hkv);
    B.post_conv = f32("attn_post_conv.weight", sconv_len_, n_embd_);
    B.ffn_post_conv = f32("ffn_post_conv.weight", sconv_len_, n_embd_);

    // The file's bias is [d_rel, extent] - ggml's ne0 is the projection's width - so a distance's row of
    // d_rel taps is already one contiguous run and only the pad row ggml_pad() adds has to be appended.
    // The reference pads its rows by one so that a key further off than the extent reads a zero row: keep
    // that pad, so the attention needs no bounds test on a distance.
    {
        const TensorInfo* rb = tensor(il, "attn_rel_bias.weight");
        const int64_t extent = B.swa ? window_ : rel_extent_;
        if (!rb || rb->type != kTypeF32 || (int64_t) rb->shape.size() != 2 || (int64_t) rb->shape[0] != d_rel_ ||
            (int64_t) rb->shape[1] != extent)
            throw std::runtime_error(blk(il, "attn_rel_bias.weight") + " is " + (rb ? shape_of(*rb) : "missing") +
                                     ", expected F32 [" + std::to_string(d_rel_) + ", " + std::to_string(extent) +
                                     "]");
        home(*rb).payload(*rb);
        const float* src = (const float*) home(*rb).data(*rb);
        B.bias_keys = (int) std::min<int64_t>(extent, opt_.max_context) + 1;
        std::vector<float> padded((size_t) d_rel_ * (size_t) B.bias_keys, 0.f);
        std::memcpy(padded.data(), src, (size_t) d_rel_ * (size_t) extent * sizeof(float));
        void* pad = dmalloc(padded.size() * sizeof(float), blk(il, "attn_rel_bias.weight").c_str());
        ck(cudaMemcpy(pad, padded.data(), padded.size() * sizeof(float), cudaMemcpyHostToDevice),
           "attn_rel_bias.weight");
        B.rel_bias = Dev{pad, 0, (int) B.bias_keys, (int) d_rel_, 1.f};
    }

    if (B.swa) {
        B.attn_norm = f32("attn_norm.weight", n_embd_);
        if (have("attn_k_conv_proj.weight") || have("attn_v_conv_proj.weight") || have("attn_q_inp_norm.weight") ||
            have("attn_k_inp_norm.weight") || have("attn_v_inp_norm.weight"))
            throw std::runtime_error(blk(il, "attn_k_conv_proj.weight") +
                                     ": a window layer has a full-context layer's tensors");
    } else {
        B.q_inp_norm = f32("attn_q_inp_norm.weight", n_embd_);
        B.k_inp_norm = f32("attn_k_inp_norm.weight", n_embd_);
        B.v_inp_norm = f32("attn_v_inp_norm.weight", n_embd_);
        B.k_conv_proj = mat("attn_k_conv_proj.weight", n_embd_, (int64_t) sconv_len_ * n_kv_);
        B.v_conv_proj = mat("attn_v_conv_proj.weight", n_embd_, (int64_t) sconv_len_ * n_kv_);
        if (have("attn_norm.weight"))
            throw std::runtime_error(blk(il, "attn_norm.weight") + ": a full-context layer has a window layer's norm");
    }

    // block attention residuals: the first layer of a block starts a new stream and has no query
    if (il > 0) B.attn_res_query = f32("attn_res_query.weight", n_embd_, B.swa ? 1 : 3);
    B.ffn_res_query = f32("ffn_res_query.weight", n_embd_);

    // the looped core's controllers, shared by every pass over the layer
    if (il >= core_start_ && il < core_end_) {
        Loop& lp = loops_[(size_t) (il - core_start_)];
        lp.msg_query = f32("loop_msg_query.weight", n_embd_, kCtrl);
        for (int c = 0; c < kCtrl; ++c) {
            const std::string d = std::string("loop_") + kCtrlName[c] + "_ctrl_down.weight";
            const std::string g = std::string("loop_") + kCtrlName[c] + "_ctrl_gain.weight";
            const std::string f = std::string("loop_") + kCtrlName[c] + "_ctrl_forget.weight";
            lp.ctrl_down[c] = f32(d.c_str(), 2 * (int64_t) n_embd_, gate_rank_);
            lp.ctrl_gain[c] = f32(g.c_str(), gate_rank_, n_embd_);
            lp.ctrl_forget[c] = f32(f.c_str(), gate_rank_, n_embd_);
        }
    }

    // MoE: the router, and the routed experts' places in the file, where they stay
    load_moe(il, false);
    if (!L.moe) throw std::runtime_error(blk(il, "ffn_gate_up_exps.weight") + ": every layer of this model has experts");
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {   // llama.cpp reads it when the file has it
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }

    // the model's output tensors: the first pass sizes the dense arena, the second fills it
    if (il == n_layer_ - 1 && lo_ == 0) load_geometry_tail();
}

// The output's own projection: the reference mixes the blocks one last time before its norm.  These are the
// model's, not a layer's, and upload() needs the dense arena, which Engine::load() sizes after load_geometry():
// the last layer of the first pass is where the engine has it.
void Inkling2::load_geometry_tail() {
    tok_norm_ = upload(gg_->find("token_embd_norm.weight"), "token_embd_norm.weight", n_embd_, 1, kF32);
    out_res_query_ = upload(gg_->find("output_res_query.weight"), "output_res_query.weight", n_embd_, 1, kF32);
}

// Each of the 48 visits keeps the last sconv_len_ - 1 rows of K, V, the attention's update and the FFN's
// update, at its own place in the recurrent state.
size_t Inkling2::recurrent_floats() const {
    return steps_.size() * (size_t) (sconv_len_ - 1) * (size_t) past_channels();
}

// A full-context visit keeps a row a position; a windowed one a ring of the rows a call can read.  The caches
// hang off the visits, not the blocks: a core layer is visited three times over the same weights.
void Inkling2::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t elems = (size_t) n_kv_ * (size_t) head_dim_;
    const size_t scale_row = (size_t) n_kv_ * ((size_t) head_dim_ / 32) * sizeof(uint16_t);
    const size_t f16_row = elems * sizeof(uint16_t);
    step_keys_.assign(steps_.size(), nullptr);
    step_vals_.assign(steps_.size(), nullptr);
    step_k_codes_.assign(steps_.size(), nullptr);
    step_k_scales_.assign(steps_.size(), nullptr);
    step_v_codes_.assign(steps_.size(), nullptr);
    step_v_scales_.assign(steps_.size(), nullptr);
    if (q8) std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", elems + scale_row);
    for (size_t s = 0; s < steps_.size(); ++s) {
        if (steps_[s].swa) {
            if (q8) {
                step_k_codes_[s] = (int8_t*) seq_alloc((size_t) ring_ * elems, "the attention cache");
                step_k_scales_[s] = (uint16_t*) seq_alloc((size_t) ring_ * scale_row, "the attention cache");
                step_v_codes_[s] = (int8_t*) seq_alloc((size_t) ring_ * elems, "the attention cache");
                step_v_scales_[s] = (uint16_t*) seq_alloc((size_t) ring_ * scale_row, "the attention cache");
            } else {
                step_keys_[s] = (uint16_t*) seq_alloc((size_t) ring_ * f16_row, "the attention cache");
                step_vals_[s] = (uint16_t*) seq_alloc((size_t) ring_ * f16_row, "the attention cache");
            }
        } else if (q8) {
            step_k_codes_[s] = (int8_t*) kv_alloc(elems, 1);
            step_k_scales_[s] = (uint16_t*) kv_alloc(scale_row, 1);
            step_v_codes_[s] = (int8_t*) kv_alloc(elems, 1);
            step_v_scales_[s] = (uint16_t*) kv_alloc(scale_row, 1);
        } else {
            step_keys_[s] = (uint16_t*) kv_alloc(f16_row, 1);
            step_vals_[s] = (uint16_t*) kv_alloc(f16_row, 1);
        }
    }
}

// Per-token buffers hold rows_ tokens, row after row.
void Inkling2::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    const size_t rel = n * n_head_ * d_rel_, nk = (size_t) max_keys_;
    qraw_ = fbuf(heads);
    kraw_ = fbuf(kv);
    vraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    vn_ = fbuf(kv);
    rraw_ = fbuf(rel);
    att_ = fbuf(heads);
    gate_ = fbuf(heads);
    sinks_ = fbuf(n * n_head_);
    tau_ = fbuf(n);
    kqb_ = fbuf(n * n_head_ * nk);
    dyn_ = fbuf(n * (size_t) sconv_len_ * n_kv_);
    conv_ = fbuf(n * (size_t) std::max(2 * n_kv_ * head_dim_, n_embd_));
    xin_ = fbuf(n * 3 * n_embd_);
    mix_in_ = fbuf(n * 3 * n_embd_);
    msg_ = fbuf(n * kCtrl * n_embd_);
    ctrl_ = fbuf(n * ((size_t) gate_rank_ + 2 * n_embd_));
    states_ = fbuf(n * (size_t) max_states_ * n_embd_);
    bank_ = fbuf((size_t) max_states_ * n_embd_ * n);
    prev_bank_ = fbuf((size_t) max_states_ * n_embd_ * n);
    prev_out_ = fbuf(n * n_embd_);
    pass_in_ = fbuf(n * n_embd_);
    // flash-decoding's splits of a decoding step's keys (see ink_attend_rel_split); the reduce is cheap
    // enough that this is always resident rather than allocated on demand
    split_out_ = fbuf((size_t) kAttnSplits * n_head_ * head_dim_);
    split_ml_ = fbuf((size_t) kAttnSplits * n_head_ * 2);

    // Each windowed visit's rows sit after the recurrent state, in the order the visits come.
    step_extra_at_.assign(steps_.size(), 0);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t side_row = q8 ? (size_t) n_kv_ * head_dim_ + (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t)
                               : (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    const size_t ring_bytes = 2 * (size_t) keep_ * side_row;
    size_t at = 0;
    for (size_t s = 0; s < steps_.size(); ++s) {
        if (!steps_[s].swa) continue;
        step_extra_at_[s] = at;
        at += ring_bytes;
    }
}

// A sequence's rows are the full-context visits': a windowed visit's are its state (save_layer()).
size_t Inkling2::card_row_bytes() const {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t side_row = q8 ? (size_t) n_kv_ * head_dim_ + (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t)
                               : (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    size_t full = 0;
    for (const Step& s : steps_)
        if (!s.swa) ++full;
    return full * 2 * side_row;
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t c_row = (size_t) n_kv_ * head_dim_ * (q8 ? size_t(1) : sizeof(uint16_t));
    const size_t s_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
    const size_t at_c = (size_t) first * c_row, n_c = (size_t) n * c_row;
    const size_t at_s = (size_t) first * s_row, n_s = (size_t) n * s_row;
    for (size_t s = 0; s < steps_.size(); ++s) {
        if (steps_[s].swa || n == 0) continue;
        if (q8) {
            ck(cudaMemcpy(dst, (const uint8_t*) step_k_codes_[s] + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) step_k_scales_[s] + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            ck(cudaMemcpy(dst, (const uint8_t*) step_v_codes_[s] + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) step_v_scales_[s] + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            continue;
        }
        for (const uint16_t* cache : {step_keys_[s], step_vals_[s]}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
        }
    }
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t c_row = (size_t) n_kv_ * head_dim_ * (q8 ? size_t(1) : sizeof(uint16_t));
    const size_t s_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
    const size_t at_c = (size_t) first * c_row, n_c = (size_t) n * c_row;
    const size_t sk_c = (size_t) (first - from) * c_row, all_c = (size_t) count * c_row;
    const size_t at_s = (size_t) first * s_row, n_s = (size_t) n * s_row;
    const size_t sk_s = (size_t) (first - from) * s_row, all_s = (size_t) count * s_row;
    for (size_t s = 0; s < steps_.size(); ++s) {
        if (steps_[s].swa || n == 0) continue;
        if (q8) {
            ck(cudaMemcpy((uint8_t*) step_k_codes_[s] + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) step_k_scales_[s] + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            ck(cudaMemcpy((uint8_t*) step_v_codes_[s] + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) step_v_scales_[s] + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            continue;
        }
        for (uint16_t* cache : {step_keys_[s], step_vals_[s]}) {
            ck(cudaMemcpy((uint8_t*) cache + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
        }
    }
}

// What a windowed visit has of the sequence: the rows the next token attends, kept at its place in the
// recurrent state.  A full-context visit's rows are save_card_rows()'.
size_t Inkling2::extra_bytes() const {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t side_row = q8 ? (size_t) n_kv_ * head_dim_ + (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t)
                               : (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    size_t swa = 0;
    for (const Step& s : steps_)
        if (s.swa) ++swa;
    return swa * 2 * (size_t) keep_ * side_row;
}

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).  A position before the sequence's start has no run.
namespace {
template <class F>
void window_runs(int64_t pos, int64_t keep, int64_t ring, F&& f) {
    for (int64_t p = std::max<int64_t>(0, pos - keep); p < pos;) {
        const int64_t at = p % ring, m = std::min<int64_t>(pos - p, ring - at);
        f((size_t) (p - (pos - keep)), (size_t) at, (size_t) m);
        p += m;
    }
}
}  // namespace

void Inkling2::save_layer(int il, int64_t pos, uint8_t* dst) {
    (void) il;
    if (!steps_[(size_t) cur_step_].swa) return;
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t c_row = (size_t) n_kv_ * head_dim_ * (q8 ? size_t(1) : sizeof(uint16_t));
    const size_t s_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
    dst += state_n_ * sizeof(float) + step_extra_at_[(size_t) cur_step_];
    auto save_ring = [&](const uint8_t* cache, size_t row) {
        std::memset(dst, 0, (size_t) keep_ * row);   // the places of positions before the sequence's start
        window_runs(pos, keep_, ring_, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpy(dst + slot * row, cache + at * row, m * row, cudaMemcpyDeviceToHost), "save state");
        });
        dst += (size_t) keep_ * row;
    };
    if (q8) {
        save_ring((const uint8_t*) step_k_codes_[(size_t) cur_step_], c_row);
        save_ring((const uint8_t*) step_k_scales_[(size_t) cur_step_], s_row);
        save_ring((const uint8_t*) step_v_codes_[(size_t) cur_step_], c_row);
        save_ring((const uint8_t*) step_v_scales_[(size_t) cur_step_], s_row);
        return;
    }
    save_ring((const uint8_t*) step_keys_[(size_t) cur_step_], c_row);
    save_ring((const uint8_t*) step_vals_[(size_t) cur_step_], c_row);
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t c_row = (size_t) n_kv_ * head_dim_ * (q8 ? size_t(1) : sizeof(uint16_t));
    const size_t s_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
    for (size_t s = 0; s < steps_.size(); ++s) {
        if (!steps_[s].swa) continue;
        auto load_ring = [&](uint8_t* cache, size_t row) {
            window_runs(pos, keep_, ring_, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy(cache + at * row, src + slot * row, m * row, cudaMemcpyHostToDevice), "load state");
            });
            src += (size_t) keep_ * row;
        };
        if (q8) {
            load_ring((uint8_t*) step_k_codes_[s], c_row);
            load_ring((uint8_t*) step_k_scales_[s], s_row);
            load_ring((uint8_t*) step_v_codes_[s], c_row);
            load_ring((uint8_t*) step_v_scales_[s], s_row);
            continue;
        }
        load_ring((uint8_t*) step_keys_[s], c_row);
        load_ring((uint8_t*) step_vals_[s], c_row);
    }
}

int Inkling2::next_moe(int il) const {
    (void) il;
    const size_t s = (size_t) cur_step_ + 1;
    return s < steps_.size() ? steps_[s].phys : n_layer_;
}

// The bank of block inputs a block's first layer starts: the value the reference's `blocks` vector holds.
void Inkling2::push_bank(const float* x, int n) {
    if (n_bank_ >= max_states_)
        throw std::logic_error("inkling2: the depth attention's bank is full (" + std::to_string(n_bank_) + " of " +
                              std::to_string(max_states_) + " at step " + std::to_string(cur_step_) + ")");
    ck(cudaMemcpyAsync(bank_ + (size_t) n_bank_ * n_embd_ * (size_t) rows_, x,
                        (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
       "bank");
    ++n_bank_;
}

// The depth attention (`mix()`): the bank's blocks newest first, then `cur`, which the reference's concat
// leaves last; every state is RMS-normed without a gain and each of the n_q queries mixes them.
void Inkling2::mix_depth(const float* bank, int n_bank, const float* cur, const float* query, int n_q, float* out,
                         int n) {
    ink_bank_states(bank, cur, states_, n_embd_, n_bank, rows_ * n_embd_, 0, n, st_);
    ink_depth_mix(states_, query, out, n_embd_, n_bank + 1, n_q, n, st_);
}

// A window layer's depth attention has one query, so its one mix is the input of all three streams: copy it
// into the other two blocks.
void Inkling2::mix_depth_copy(const float* x, int n, float* out) {
    if (x == out) return;
    ck(cudaMemcpyAsync(out, x, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
       "depth mix copy");
}

// The four messages a pass hands one core layer: the pass before's blocks after the first, and its output.
// The query is the LAYER's own - the reference computes `mix(bank, bank_out, loop.msg_query)` inside the
// `for ip` loop with `loop_layers[ip - start]`, so every core layer asks the same bank a different question.
void Inkling2::pass_messages(int il, int n) {
    const Loop& lp = loops_[(size_t) (il - core_start_)];
    ink_bank_states(prev_bank_, prev_out_, states_, n_embd_, n_prev_bank_, rows_ * n_embd_, 0, n, st_);
    ink_depth_mix(states_, fp(lp.msg_query), msg_, n_embd_, n_prev_bank_ + 1, kCtrl, n, st_);
    dbg("msg", il, msg_, n * n_embd_ * kCtrl);
}

// A visit's K or V into its cache: a full-context one at the position, a windowed one at its place in the ring.
void Inkling2::store(const float* x, int s, int64_t pos, int n, bool value) {
    const int width = n_kv_ * head_dim_;
    if (opt_.kv_format == KvFmt::kQ8_0) {
        int8_t* codes = value ? step_v_codes_[(size_t) s] : step_k_codes_[(size_t) s];
        uint16_t* scales = value ? step_v_scales_[(size_t) s] : step_k_scales_[(size_t) s];
        if (!steps_[(size_t) s].swa) return store_q8(x, codes, scales, width, pos, n, st_);
        const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
        store_q8(x, codes, scales, width, at, head, st_);
        if (head < n) store_q8(x + (size_t) head * width, codes, scales, width, 0, n - head, st_);
        return;
    }
    uint16_t* cache = value ? step_vals_[(size_t) s] : step_keys_[(size_t) s];
    if (!steps_[(size_t) s].swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

// The attention: no rotary position, a learned relative-position bias that the queries and their projection
// both feed, and the values normed without a gain.  The K and V go through a causal short convolution of the
// visit's own history first, and the output through one more after the gate.
void Inkling2::attn(Block& B, int il, int n) {
    const Step& st = steps_[(size_t) cur_step_];
    const int hd = head_dim_, nh = n_head_, nkv = n_kv_;
    const int chan_kv = nkv * hd;
    const int s = cur_step_;
    const bool shared_in = B.swa && st.pass == 0;
    const float* in[3] = {xin_, shared_in ? xin_ : xin_ + (size_t) n * n_embd_,
                          shared_in ? xin_ : xin_ + (size_t) 2 * n * n_embd_};

    // Quantize in[0] to Q8_1 once and run every in[0] projection first (wq, rel, w_gate, sink_proj) so wk/wv
    // (which overwrite xq_ when shared_in is false) cannot clobber in[0]'s quantized buffer.
    kernels::native_quantize_q8_1(in[0], xq_, n_embd_ * n, 1, st_);
    lin(B.wq, in[0], qraw_, n, xq_);
    lin(B.rel, in[0], rraw_, n, xq_);
    lin(B.w_gate, in[0], gate_, n, xq_);
    lin(B.sink_proj, in[0], sinks_, n, xq_);
    lin(B.wk, in[1], kraw_, n, shared_in ? xq_ : nullptr);
    lin(B.wv, in[2], vraw_, n, shared_in ? xq_ : nullptr);

    // the short convolutions of K and V, each at its own place in the visit's history
    if (B.swa) {
        ink_conv(kraw_, past(s) + past_at(0), fp(B.k_conv), nullptr, k_, chan_kv, n, sconv_len_, nkv, st_);
        ink_conv(vraw_, past(s) + past_at(1), fp(B.v_conv), nullptr, v_, chan_kv, n, sconv_len_, nkv, st_);
    } else {
        lin(B.k_conv_proj, in[1], dyn_, n, nullptr);
        ink_conv(kraw_, past(s) + past_at(0), fp(B.k_conv), dyn_, k_, chan_kv, n, sconv_len_, nkv, st_);
        lin(B.v_conv_proj, in[2], dyn_, n, nullptr);
        ink_conv(vraw_, past(s) + past_at(1), fp(B.v_conv), dyn_, v_, chan_kv, n, sconv_len_, nkv, st_);
    }

    // clip() and the values' norm, then the log-scaled positions of a full-context layer (after the clip)
    ink_clip_l2(qraw_, fp(B.q_clip), hd, nh, n, qk_clip_, st_);
    ink_clip_l2(k_, nullptr, hd, nkv, n, qk_clip_, st_);
    rms_norm(v_, nullptr, vn_, hd, n * nkv, rms_eps_, st_);
    ink_clip_l2(rraw_, fp(B.rel_clip), d_rel_, nh, n, rel_clip_, st_);
    dbg("q_clip", il, qraw_, n * nh * hd);
    dbg("kraw", il, kraw_, n * nkv * hd);
    dbg("vraw", il, vraw_, n * nkv * hd);
    dbg("k_conv", il, k_, n * nkv * hd);
    dbg("vn", il, vn_, n * nkv * hd);
    dbg("r_clip", il, rraw_, n * nh * d_rel_);
    // The log-scaled positions of a full-context layer: the factor multiplies the queries' own q.k and the
    // relative projection's bias, so it goes to the two kernels rather than onto qraw_/rraw_ here - the reference
    // scales Qcur and Rcur once and the logit is then (q*tau).k and bias.(r*tau).
    const float* tau = nullptr;
    if (!B.swa) {
        ink_tau(tau_, pos_, log_scale_floor_, log_scale_alpha_, n, st_);
        tau = tau_;
    }

    // the keys and values this token contributes, into the visit's cache (a ring, for a window layer)
    store(k_, s, pos_, n, false);
    store(vn_, s, pos_, n, true);

    // the bias by distance, with the sink folded in (sinks_ computed above from in[0]), and then the attention
    ink_rel_table(fp(B.rel_bias), d_rel_, rraw_, sinks_, tau, kqb_, n, nh, B.bias_keys, st_);
    dbg("sinks", il, sinks_, n * nh);
    dbg("kqb", il, kqb_, n * nh * B.bias_keys);
    if (cur_step_ == 0 && il == 0)
        for (int h = 0; h < 4; ++h) {
            char nm[32];
            std::snprintf(nm, sizeof nm, "kqb_h%d", h);
            dbg(nm, il, kqb_ + (size_t) h * B.bias_keys, 3);
        }
    // A decoding step over a full-context layer's long key range: the one-block kernel launches a block a
    // head, 24 blocks, and each walks the whole range serially, so the keys are split across kAttnSplits
    // more and the reduce folds the splits back (flash decoding).  Below kAttnSplitMin the range is short
    // enough that the extra launches and the reduce cost more than the idle blocks do.
    const bool split = n == 1 && !B.swa && pos_ >= kAttnSplitMin;
    const float ascale = 1.f / std::sqrt((float) hd);
    if (opt_.kv_format == KvFmt::kQ8_0) {
        if (split)
            ink_attend_rel_q8_split(qraw_, step_k_codes_[(size_t) s], step_k_scales_[(size_t) s],
                                     step_v_codes_[(size_t) s], step_v_scales_[(size_t) s], kqb_, tau, pos_, att_,
                                     split_out_, split_ml_, hd, nh, nkv, ascale, 0, B.bias_keys, 0, kAttnSplits, st_);
        else
            ink_attend_rel_q8(qraw_, step_k_codes_[(size_t) s], step_k_scales_[(size_t) s],
                              step_v_codes_[(size_t) s], step_v_scales_[(size_t) s], kqb_, tau, pos_, n, att_, hd, nh,
                              nkv, ascale, B.swa ? window_ : 0, B.bias_keys, B.swa ? ring_ : 0, st_);
    } else {
        if (split)
            ink_attend_rel_split(qraw_, step_keys_[(size_t) s], step_vals_[(size_t) s], kqb_, tau, pos_, att_, split_out_,
                                  split_ml_, hd, nh, nkv, ascale, 0, B.bias_keys, 0, kAttnSplits, st_);
        else
            ink_attend_rel(qraw_, step_keys_[(size_t) s], step_vals_[(size_t) s], kqb_, tau, pos_, n, att_, hd, nh, nkv,
                            ascale, B.swa ? window_ : 0, B.bias_keys, B.swa ? ring_ : 0, st_);
    }
    dbg("att_raw", il, att_, n * nh * hd);
    if (cur_step_ == 0 && il == 0)
        for (int h = 0; h < 4; ++h) {
            char nm[32];
            std::snprintf(nm, sizeof nm, "att_h%d", h);
            dbg(nm, il, att_ + (size_t) h * hd, hd);
        }

    // the gate on the attention's output (gate_ computed above from in[0]), and its own short convolution.
    dbg("gate", il, gate_, n * nh * hd);
    mul_sigmoid(att_, gate_, hd, n * nh, hd, false, st_);
    dbg("att_gated", il, att_, n * nh * hd);
    lin(B.wo, att_, conv_, n, nullptr);
    dbg("conv_", il, conv_, n * n_embd_);
    ink_conv(conv_, past(s) + past_at(2), fp(B.post_conv), nullptr, ao_, n_embd_, n, sconv_len_, 1, st_);
    dbg("ao", il, ao_, n * n_embd_);
}

// A layer from its input to its router's copy.  The attention's input is the depth attention's mix of the
// earlier blocks' inputs, and the FFN's the same mix of the blocks after this layer's input joined them.
void Inkling2::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const Step& st = steps_[(size_t) cur_step_];
    const int n_q = B.swa ? 1 : 3;

    // The embedding is normed before the first layer, as the reference's build_inputs() does.  A call starts
    // here, so the depth attention's bank and the pass before's are empty again (begin_pass() runs at load).
    if (cur_step_ == 0) {
        n_bank_ = 0;
        n_prev_bank_ = 0;
    }
    if (cur_step_ == 0) rms_norm(r_, fp(tok_norm_), r_, n_embd_, n, rms_eps_, st_);
    dbg("tok", 0, r_, n * n_embd_);

    // every pass starts from the prelude's output, and each of its core layers reads the pass before's
    // blocks through its own message query
    if (st.pass > 0 && il >= core_start_ && il < core_end_) {
        if (il == core_start_) {
            n_bank_ = n_prelude_blocks_;
            // Every pass after the first starts from the prelude's output, not from the pass before's:
            // docs/models/reference/inkling2-loop.cpp sets `cur = prelude` and `blocks = prelude_blocks`.
            ck(cudaMemcpyAsync(r_, pass_in_, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
               "pass");
        }
        pass_messages(il, n);
    }

    // The attention's input: the depth attention mixes `cur` with the earlier blocks' inputs.  A full-context
    // layer takes three mixes (Q, K, V); a window layer one, which all three of its inputs share until the
    // messages part them (on pass 0 they stay identical, so norm once into xin_ and skip the two copies).
    const bool shared_in = B.swa && st.pass == 0;
    if (il > 0) {
        mix_depth(bank_, n_bank_, r_, fp(B.attn_res_query), n_q, mix_in_, n);
        if (n_q == 1 && !shared_in) {
            mix_depth_copy(mix_in_, n, mix_in_ + (size_t) n * n_embd_);
            mix_depth_copy(mix_in_, n, mix_in_ + (size_t) 2 * n * n_embd_);
        }
    } else if (!shared_in) {
        for (int j = 0; j < 3; ++j) mix_depth_copy(r_, n, mix_in_ + (size_t) j * n * n_embd_);
    }
    if (st.pass > 0) {
        const Loop& lp = loops_[(size_t) (st.phys - core_start_)];
        for (int j = 0; j < 3; ++j)
            ink_control(mix_in_ + (size_t) j * n * n_embd_, msg_ + (size_t) j * n * n_embd_, fp(lp.ctrl_down[j]),
                         fp(lp.ctrl_gain[j]), fp(lp.ctrl_forget[j]), n_embd_, gate_rank_, n, ctrl_, st_);
    }
    // a window layer normed once shares the one input with all three, as the reference does
    if (shared_in) {
        rms_norm(il > 0 ? mix_in_ : r_, fp(B.attn_norm), xin_, n_embd_, n, rms_eps_, st_);
    } else if (B.swa) {
        for (int j = 0; j < 3; ++j)
            rms_norm(mix_in_ + (size_t) j * n * n_embd_, fp(B.attn_norm), xin_ + (size_t) j * n * n_embd_, n_embd_, n,
                      rms_eps_, st_);
    } else {
        rms_norm(mix_in_, fp(B.q_inp_norm), xin_, n_embd_, n, rms_eps_, st_);
        rms_norm(mix_in_ + (size_t) n * n_embd_, fp(B.k_inp_norm), xin_ + (size_t) n * n_embd_, n_embd_, n, rms_eps_,
                  st_);
        rms_norm(mix_in_ + (size_t) 2 * n * n_embd_, fp(B.v_inp_norm), xin_ + (size_t) 2 * n * n_embd_, n_embd_, n,
                  rms_eps_, st_);
    }
    dbg("attn_in", il, xin_, n * n_embd_);
    dbg("attn_in1", il, xin_ + (size_t) n * n_embd_, n * n_embd_);
    dbg("attn_in2", il, xin_ + (size_t) 2 * n * n_embd_, n * n_embd_);

    attn(B, il, n);
    dbg("attn", il, ao_, n * n_embd_);

    // the layer's input joins the bank after the attention has read it, and the residual either starts a
    // new stream (a block's first layer) or adds to the one it found
    if (B.block_start) push_bank(r_, n);
    if (B.block_start) {
        ck(cudaMemcpyAsync(r_, ao_, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
           "block residual");
    } else {
        axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
    }

    // the FFN's input is the depth attention's mix of the bank, now with this layer's input in it
    mix_depth(bank_, n_bank_, r_, fp(B.ffn_res_query), 1, mix_in_, n);
    if (st.pass > 0) {
        const Loop& lp = loops_[(size_t) (st.phys - core_start_)];
        ink_control(mix_in_, msg_ + (size_t) 3 * n * n_embd_, fp(lp.ctrl_down[3]), fp(lp.ctrl_gain[3]),
                     fp(lp.ctrl_forget[3]), n_embd_, gate_rank_, n, ctrl_, st_);
    }
    rms_norm(mix_in_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    dbg("ffn_in", il, xn_, n * n_embd_);
    // The router and the experts both read the quantized FFN input, so it is quantized here: lin() only quantizes
    // when the call is wide enough for its matrix kernels, and a short call would leave xq_ as it found it.
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    if (!batched(L, n)) moe_router(L, n);
}

// The routed experts' sum through the FFN's own short convolution, added to the layer's residual.
void Inkling2::layer_tail(Layer& L, int il, int n) {
    const Step& st = steps_[(size_t) cur_step_];
    ink_conv(fo_, past(cur_step_) + past_at(3), fp(blocks_[(size_t) il].ffn_post_conv), nullptr, conv_, n_embd_, n,
              sconv_len_, 1, st_);
    axpy(conv_, 1.f, r_, n * n_embd_, true, st_);
    dbg("l_out", il, r_, n * n_embd_);

    // the last prelude layer's output: every pass after the first starts from it
    if (st.phys == core_start_ - 1)
        ck(cudaMemcpyAsync(pass_in_, r_, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
           "pass");

    // the last core layer of a pass: the next pass's messages read this one's blocks and output
    if (st.pass < loop_count_ - 1 && st.phys == core_end_ - 1) {
        n_prev_bank_ = n_bank_ - (n_prelude_blocks_ + 1);
        if (n_prev_bank_ > 0)
            ck(cudaMemcpyAsync(prev_bank_, bank_ + (size_t) (n_prelude_blocks_ + 1) * n_embd_ * (size_t) rows_,
                               (size_t) n_prev_bank_ * n_embd_ * (size_t) rows_ * sizeof(float),
                               cudaMemcpyDeviceToDevice, cs(st_)),
               "bank");
        ck(cudaMemcpyAsync(prev_out_, r_, (size_t) rows_ * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice,
                           cs(st_)),
           "bank");
    }
}

// The model's output: the depth attention's last mix, then the output norm.
void Inkling2::output_rows(int first, int rows, float* h) {
    ink_bank_states(bank_, r_, states_, n_embd_, n_bank_, rows_ * n_embd_, first, rows, st_);
    ink_depth_mix(states_, fp(out_res_query_), mix_in_, n_embd_, n_bank_ + 1, 1, rows, st_);
    rms_norm(mix_in_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and
// weights from the bare gates, made to sum to one.
void Inkling2::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (route_scale_ != 0.f) w[k] *= route_scale_;
    }
}

bool Inkling2::route_card(int il, int n, int32_t* ids, float* w) {
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(blocks_[(size_t) il].probs_b_dev), nullptr, w_norm_,
               route_scale_, ids, w, st_);
    return true;
}

// No shared expert: the routed experts' sum starts at zero.
void Inkling2::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

// The final softcap, and the padded vocabulary the reference drops.
void Inkling2::finish_logits(float* logits, int rows) {
    ink_finish_logits(logits, n_vocab_, n_vocab_unpadded_, final_softcap_, rows, st_);
    // --dump-layers: the logits themselves, past the pad, for the cosine gate's third point.  Only the
    // one row a dump runs has them here; the kernel leaves the pad untouched, so n is the unpadded size.
    if (rows == 1) dbg("logits", n_layer_, logits, n_vocab_unpadded_);
}

}  // namespace strata::poly
