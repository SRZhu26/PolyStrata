// src/poly/inkling2.cpp - Thinking Machines Lab's Inkling 2 Mini ("inkling2_loop") scaffolding.
//
// Reference layout: ggml-org/llama.cpp PR #32694 (inkling2_loop) + PR #32704 (per-expert NVFP4 scales on looped
// passes 2 and 3), verified against Unsloth's Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file header.
//
// Status: load_geometry() and KV cache sizing (FP16 and Q8_0 across 8 global + 40 SWA logical layers) are wired;
// the architecture-specific forward kernels (relative-position attention, short conv, loop gates, attention
// residuals, and fused NVFP4 ffn_gate_up_exps with situ softcaps) are stubbed for bringup on a CUDA/HIP machine.
#include "inkling2.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size())
        throw std::runtime_error("inkling2_loop: per-layer metadata array ends before block_count");
    return (int64_t) v.items[(size_t) il].u;
}

size_t kv_row_bytes_for(int n_kv, int head_dim, bool q8) {
    const size_t vals = (size_t) n_kv * (size_t) head_dim;
    // Q8_0: 32 int8 codes + 2-byte fp16 scale per 32 values = 34 B per 32 values (per K or V side).
    return q8 ? vals + (vals / 32) * sizeof(uint16_t) : vals * sizeof(uint16_t);
}

[[noreturn]] void not_yet_implemented(const char* where) {
    throw std::runtime_error(
        std::string("inkling2_loop engine scaffolding: ") + where +
        " is not implemented yet. Remaining bringup items: (1) relative-position attention + log-scaling + "
        "QK/rel L2 clip + value norm, (2) short conv (k=4), (3) looped-core gates (rank 64) + attention residuals "
        "(block 6), (4) fused NVFP4 ffn_gate_up_exps with situ softcaps (4.0/25.0) and per-pass scales. "
        "See docs/models/inkling2.md for the bringup and llama.cpp parity checklist.");
}

}  // namespace

void Inkling2::begin_pass() {}

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

    const MetaValue& hc = must("attention.head_count");
    const MetaValue& hkv = must("attention.head_count_kv");
    n_head_ = (int) of_layer(hc, 0);
    n_kv_ = (int) of_layer(hkv, 0);
    for (int il = 1; il < n_layer_; ++il) {
        need((int) of_layer(hc, il) == n_head_, "varying attention.head_count across layers");
        need((int) of_layer(hkv, il) == n_kv_, "varying attention.head_count_kv across layers");
    }
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
    situ_beta_ = shape_f("activation.situ_beta");
    situ_linear_beta_ = shape_f("activation.situ_linear_beta");
    route_scale_ = shape_f("expert_weights_scale");
    final_softcap_ = shape_f("final_logit_softcapping");

    core_start_ = shape_i("loop.start");
    core_end_ = shape_i("loop.end");
    loop_count_ = shape_i("loop.count");
    gate_rank_ = shape_i("loop.gate_rank");
    need(core_start_ >= 0 && core_start_ < core_end_ && core_end_ <= n_layer_, "bad loop.start/loop.end");
    need(loop_count_ >= 1, "bad loop.count");

    const MetaValue& swa = must("attention.sliding_window_pattern");
    swa_pattern_.resize((size_t) n_layer_);
    for (int il = 0; il < n_layer_; ++il) swa_pattern_[(size_t) il] = of_layer(swa, il) != 0 ? 1 : 0;

    // Build the unrolled logical schedule: prelude [0, core_start_), looped core [core_start_, core_end_) x loop_count_,
    // epilogue [core_end_, n_layer_).  For Inkling 2 Mini: 6 + 12*3 + 6 = 48 logical steps (8 global, 40 local SWA).
    steps_.clear();
    n_global_steps_ = 0;
    n_local_steps_ = 0;
    auto push = [&](int phys, int pass) {
        const bool is_swa = swa_pattern_[(size_t) phys] != 0;
        steps_.push_back(Step{phys, pass, is_swa});
        if (is_swa) ++n_local_steps_;
        else ++n_global_steps_;
    };
    for (int il = 0; il < core_start_; ++il) push(il, 0);
    for (int p = 0; p < loop_count_; ++p)
        for (int il = core_start_; il < core_end_; ++il) push(il, p);
    for (int il = core_end_; il < n_layer_; ++il) push(il, 0);

    // Note: rows_ (prompt batch size) is not settled until after load_geometry() (Engine::load() sets it before
    // alloc_caches()).  Size ring_ and keep_ in alloc_caches(), matching Flint::alloc_caches().
    const bool use_q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t side = kv_row_bytes_for(n_kv_, head_dim_, use_q8);
    const size_t global_tok = (size_t) n_global_steps_ * 2 * side;
    std::fprintf(stderr,
                 "strata-poly: inkling2_loop: %d physical -> %zu logical layers (%d global, %d SWA=%d), "
                 "%s KV: %zu B/row (%zu B/tok global)\n",
                 n_layer_, steps_.size(), n_global_steps_, n_local_steps_, window_,
                 use_q8 ? "q8_0" : "fp16", side, global_tok);
}

size_t Inkling2::card_row_bytes() const {
    const bool use_q8 = opt_.kv_format == KvFmt::kQ8_0;
    return (size_t) n_global_steps_ * 2 * kv_row_bytes_for(n_kv_, head_dim_, use_q8);
}

size_t Inkling2::extra_bytes() const {
    const bool use_q8 = opt_.kv_format == KvFmt::kQ8_0;
    const int k = keep_ > 0 ? keep_ : std::min(window_, opt_.max_context);
    return (size_t) n_local_steps_ * (size_t) k * 2 * kv_row_bytes_for(n_kv_, head_dim_, use_q8);
}

void Inkling2::load_layer(int il) {
    (void) il;
    not_yet_implemented("Inkling2::load_layer()");
}

void Inkling2::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    not_yet_implemented("Inkling2::alloc_caches()");
}
void Inkling2::assign_scratch() { not_yet_implemented("Inkling2::assign_scratch()"); }

void Inkling2::layer_head(Layer& L, int il, int n) {
    (void) L;
    (void) il;
    (void) n;
    not_yet_implemented("Inkling2::layer_head()");
}

void Inkling2::layer_tail(Layer& L, int il, int n) {
    (void) L;
    (void) il;
    (void) n;
    not_yet_implemented("Inkling2::layer_tail()");
}

void Inkling2::output_rows(int first, int rows, float* h) {
    (void) first;
    (void) rows;
    (void) h;
    not_yet_implemented("Inkling2::output_rows()");
}

void Inkling2::route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const {
    (void) il;
    (void) logits;
    (void) token;
    (void) ids;
    (void) w;
    (void) tmp;
    not_yet_implemented("Inkling2::route()");
}

bool Inkling2::route_card(int il, int n, int32_t* ids, float* w) {
    (void) il;
    (void) n;
    (void) ids;
    (void) w;
    return false;
}

void Inkling2::moe_shared(const Layer& L, int il, int n) {
    (void) L;
    (void) il;
    (void) n;
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    (void) first;
    (void) n;
    (void) dst;
    not_yet_implemented("Inkling2::save_card_rows()");
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    (void) src;
    (void) from;
    (void) count;
    (void) first;
    (void) n;
    not_yet_implemented("Inkling2::load_card_rows()");
}

void Inkling2::save_layer(int il, int64_t pos, uint8_t* dst) {
    (void) il;
    (void) pos;
    (void) dst;
    not_yet_implemented("Inkling2::save_layer()");
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
    (void) src;
    (void) pos;
    not_yet_implemented("Inkling2::load_extra()");
}

int Inkling2::head_key(int il, int n) const {
    (void) il;
    (void) n;
    return 0;
}

}  // namespace strata::poly
