// src/poly/inkling2.hpp - Thinking Machines Lab's Inkling 2 Mini (llama.cpp architecture "inkling2_loop"):
// 26.2B-A5.4B.
//
// 24 physical layers where the core [core_start_, core_end_) runs loop_count_ times over the same weights with
// low-rank loop gates (48 logical layer applications a token).  20 of the 24 physical layers use a 1,024-token
// sliding window and 4 attend the full context, so the 48 visits keep 40 local rings and 8 full-context caches.
// There is no rotary position: a learned relative-position bias carries the positions, and the full-context layers
// scale theirs logarithmically.  Each layer adds the attention's and the FFN's output through a causal short
// convolution of its own history, and mixes its input with the earlier blocks' (the reference's `mix()`).
// Routed experts are stored fused as ffn_gate_up_exps + ffn_down_exps (NVFP4/Q8_0/Q5_K) with per-expert global
// scales.  The reference for the math is llama.cpp's inkling2 graph
// (docs/models/reference/inkling2.cpp and inkling2-loop.cpp, kept beside this file).
#pragma once

#include "engine.hpp"

#include <vector>

namespace strata::poly {

class Inkling2 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    int n_steps() const override { return (int) steps_.size(); }
    int step_phys(int s) const override { return steps_[(size_t) s].phys; }
    int next_moe(int il) const override;
    bool halves_ok() const override { return false; }          // until Phase 3
    /// Not yet: a layer-major read runs every layer over every call, and the depth attention's bank and the
    /// pass before's output cross a call's layers, so they would have to be staged per call.  The staging
    /// diverges at blk 6 (see branch inkling2-layer-major-wip), so this stays a call at a time.
    bool read_layer_major() const override { return false; }
    bool state_whole() const override { return true; }           // a layer's index does not name its visit's state
    size_t recurrent_floats() const override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return n_head_ * head_dim_; }
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    /// route() divides the chosen experts' weights by their sum and then applies the router's own
    /// expert_weights_scale, as the reference does; the engine's moe() must not divide them again.
    bool weights_normalized() const override { return true; }
    void finish_logits(float* logits, int rows) override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    int head_key(int il, int n) const override {
        (void) il;
        return room_key(n) * loop_count_ + steps_[(size_t) cur_step_].pass;
    }
    /// The looped core visits the same weights three times with different work after the experts (its own depth
    /// attention bank, its own pass messages, its own controllers), so a physical layer's tail is recorded once for each
    /// pass and not once for each number of tokens alone.
    int tail_key() const override { return steps_[(size_t) cur_step_].pass; }

private:
    /// One logical execution step: the prelude, then the core three times, then the epilogue.
    struct Step {
        int phys = 0;   ///< the physical layer in [0, n_layer_)
        int pass = 0;   ///< 0 in the prelude and epilogue, else 1..loop_count_-1 in the core
        bool swa = false;
        int local = -1;   ///< its place among the SWA steps, or -1 for a full-context one
    };

    /// The looped core's four rank-`gate_rank_` controllers: the message query and each controller's down,
    /// gain and forget matrices.
    struct Loop {
        Dev msg_query;          ///< [n_embd, 4]
        Dev ctrl_down[4];       ///< [2 * n_embd, gate_rank]
        Dev ctrl_gain[4];       ///< [gate_rank, n_embd]
        Dev ctrl_forget[4];     ///< [gate_rank, n_embd]
    };

    /// A physical layer's weights beside the ones every architecture has (Layer).
    struct Block {
        bool swa = false;             ///< attends a sliding window
        bool block_start = false;     ///< the first layer of an attn_res block: it starts a new residual stream
        Dev wq, wk, wv, wo, w_gate;   ///< the attention, and the gate on its output
        Dev attn_norm;                ///< a window layer's one input norm
        Dev q_inp_norm, k_inp_norm, v_inp_norm;   ///< a full-context layer's three
        Dev q_clip;                   ///< [head_dim, n_head], the per-channel scale clip() puts first
        Dev rel;                       ///< [n_embd, d_rel * n_head], the relative-position projection
        Dev rel_bias;                  ///< [d_rel, extent + 1], the bias a distance reads (the pad row is zero)
        Dev rel_clip;                  ///< [d_rel, n_head]
        Dev sink_proj;                 ///< [n_embd, n_head]
        Dev k_conv, v_conv;            ///< [n_conv, n_kv * head_dim]
        Dev post_conv;                 ///< [n_conv, n_embd]
        Dev ffn_post_conv;             ///< [n_conv, n_embd], the FFN's update
        Dev k_conv_proj, v_conv_proj;   ///< a full-context layer's input-dependent taps [n_embd, n_conv * n_kv]
        Dev attn_res_query;            ///< the depth attention's queries [n_embd, swa ? 1 : 3], absent at layer 0
        Dev ffn_res_query;             ///< [n_embd]
        int bias_keys = 0;             ///< the columns of rel_bias: the extent and the pad row
        std::vector<float> probs_b;     ///< the bias the routed experts' pick adds to the gates
        Dev probs_b_dev;
    };

    /// The conv history of step `s`: `sconv_len_ - 1` rows of K, then V, then the attention's update, then the
    /// FFN's, each row `past_channels()` floats.
    float* past(int s) const;
    int past_channels() const { return 2 * n_kv_ * head_dim_ + 2 * n_embd_; }
    int past_at(int which) const;

    /// The depth attention's bank of block inputs for this pass, and the pass before's.
    void push_bank(const float* x, int n);
    void mix_depth(const float* bank, int n_bank, const float* cur, const float* query, int n_q, float* out, int n);
    void mix_depth_copy(const float* x, int n, float* out);
    void pass_messages(int il, int n);
    /// The output's own projection and norm, which are not a layer's.
    void load_geometry_tail();

    void attn(Block& B, int il, int n);
    void store(const float* x, int s, int64_t pos, int n, bool value);
    int n_keys(const Block& B) const { return B.bias_keys; }

    std::vector<Block> blocks_;   ///< one a physical layer of layers_
    std::vector<Loop> loops_;     ///< one a core physical layer, in [core_start_, core_end_)
    std::vector<Step> steps_;
    std::vector<uint8_t> swa_pattern_;   ///< per physical layer: 1 = local SWA, 0 = full context

    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0;
    int d_rel_ = 0, rel_extent_ = 0, max_keys_ = 0;
    int window_ = 0, ring_ = 0, keep_ = 0;
    int sconv_len_ = 0, attn_res_block_ = 0;
    int core_start_ = 0, core_end_ = 0, loop_count_ = 1, gate_rank_ = 0;
    int n_prelude_blocks_ = 0, n_core_blocks_ = 0, n_epilogue_blocks_ = 0;
    int log_scale_floor_ = 0;
    float log_scale_alpha_ = 0.f;
    float qk_clip_ = 0.f, rel_clip_ = 0.f;
    float route_scale_ = 0.f, final_softcap_ = 0.f;
    bool w_norm_ = true;   ///< the picked experts' weights are made to sum to one
    int n_vocab_unpadded_ = 0;
    Dev tok_norm_, out_res_query_;

    /// The block inputs of the pass under way, and of the pass before it (its core blocks after the first,
    /// and its output).  The prelude's output is kept beside them: every pass after the first starts from it,
    /// not from the pass before's output (docs/models/reference/inkling2-loop.cpp: `cur = prelude`).
    float* bank_ = nullptr;
    float* prev_bank_ = nullptr;
    float* prev_out_ = nullptr;
    float* pass_in_ = nullptr;
    int n_bank_ = 0, n_prev_bank_ = 0;
    int max_states_ = 0;
    /// The place each visit's windowed rows take in the extra state: 0 for a full-context visit.
    std::vector<size_t> step_extra_at_;

    /// Per visit: the K and V caches (a full-context layer's own, a window layer's a ring of its own).
    std::vector<uint16_t*> step_keys_, step_vals_;
    /// --kv q8_0: per-visit int8 codes (n_kv * head_dim) and fp16 block scales (n_kv * head_dim / 32).
    std::vector<int8_t*> step_k_codes_, step_v_codes_;
    std::vector<uint16_t*> step_k_scales_, step_v_scales_;

    // device scratch
    float* qraw_ = nullptr;   ///< n * n_head * head_dim
    float* kraw_ = nullptr;   ///< n * n_kv * head_dim
    float* vraw_ = nullptr;
    float* k_ = nullptr;      ///< the K convolution's output, then clipped
    float* v_ = nullptr;      ///< the V convolution's output, then RMS-normed
    float* rraw_ = nullptr;   ///< n * n_head * d_rel, then clipped
    float* att_ = nullptr;    ///< n * n_head * head_dim
    float* gate_ = nullptr;   ///< the attention gate, n * n_head * head_dim
    float* sinks_ = nullptr;   ///< n * n_head
    float* kqb_ = nullptr;    ///< n * n_head * max_keys_, the relative-position bias of a token, head and distance
    float* tau_ = nullptr;    ///< n, the log-scaled position factor of a full-context layer
    float* dyn_ = nullptr;    ///< n * sconv_len_ * n_kv, a full-context layer's input-dependent taps
    float* conv_ = nullptr;    ///< a convolution's output, max(2 * n_kv * head_dim, n_embd) a token
    float* vn_ = nullptr;      ///< the values after their RMS norm, n * n_kv * head_dim
    float* xin_ = nullptr;     ///< the attention's normed inputs, 3 blocks of n * n_embd (a window layer reads the first)
    float* mix_in_ = nullptr;  ///< the depth attention's output, 3 blocks of n * n_embd
    float* msg_ = nullptr;    ///< the looped core's four messages, 4 * n_embd a token
    float* ctrl_ = nullptr;   ///< the controllers' scratch, gate_rank_ + 2 * n_embd a token
    float* states_ = nullptr;  ///< the depth attention's stacked states, max_states_ * n_embd a token
    /// Flash-decoding's split of one row's keys, for a full-context layer at kSplitMin or beyond: the splits'
    /// unnormalized value sums (kAttnSplits * n_head * head_dim) and their (largest logit, denominator) pairs.
    float* split_out_ = nullptr;
    float* split_ml_ = nullptr;
    /// Splits a decoding step's keys into, when the one-block kernel's n_head blocks would leave the card idle.
    static constexpr int kAttnSplits = 16;
    /// The position a split starts at: below it the key range is too short for the reduce to pay for itself.
    static constexpr int kAttnSplitMin = 2048;
};

}  // namespace strata::poly
