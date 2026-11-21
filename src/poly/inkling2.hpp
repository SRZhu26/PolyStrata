// src/poly/inkling2.hpp - Thinking Machines Lab's Inkling 2 Mini (llama.cpp architecture "inkling2_loop"): 26.2B-A5.4B.
//
// Scaffolding for strata-poly: 24 physical layers where core layers [core_start_, core_end_) run loop_count_ times
// with shared weights and low-rank loop gates (48 logical layer applications per token).  20 of the 24 physical
// layers use a 1,024-token sliding window and 4 attend the full context, giving 40 local SWA + 8 global KV caches
// across the 48 logical steps.  Routed experts are stored fused as ffn_gate_up_exps + ffn_down_exps (NVFP4/Q8_0/Q5_K)
// with per-expert global scales.
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
    int next_moe(int il) const override {
        (void) il;
        const size_t s = (size_t) cur_step_ + 1;
        return s < steps_.size() ? steps_[s].phys : n_layer_;
    }
    bool halves_ok() const override { return false; }   // until Phase 3
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
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    int head_key(int il, int n) const override;

private:
    /// One logical execution step (48 steps from 24 physical layers: 0..5, 6..17 x3, 18..23).
    struct Step {
        int phys = 0;   ///< physical layer index in [0, n_layer_)
        int pass = 0;   ///< 0 for prelude/epilogue; 0..loop_count_-1 inside the looped core
        bool swa = false;
    };

    std::vector<Step> steps_;
    std::vector<uint8_t> swa_pattern_;   ///< per physical layer (1 = local SWA, 0 = global full context)

    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0;
    int d_rel_ = 0, rel_extent_ = 0;
    int window_ = 0, ring_ = 0, keep_ = 0;
    int sconv_len_ = 0, attn_res_block_ = 0;
    int core_start_ = 0, core_end_ = 0, loop_count_ = 1, gate_rank_ = 0;
    int log_scale_floor_ = 0;
    float log_scale_alpha_ = 0.f;
    float qk_clip_ = 0.f, rel_clip_ = 0.f;
    float situ_beta_ = 0.f, situ_linear_beta_ = 0.f;
    float route_scale_ = 1.f, final_softcap_ = 0.f;
    int n_global_steps_ = 0, n_local_steps_ = 0;
};

}  // namespace strata::poly
