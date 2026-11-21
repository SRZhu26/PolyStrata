# Inkling 2 Mini 26B-A5B (`inkling2_loop`) — Implementation, Optimization & Test Plan

Self-contained engineering handoff for bringing up, optimizing, and verifying **Thinking Machines Lab's Inkling 2 Mini**
(`general.architecture = "inkling2_loop"`) in PolyStrata's `strata-poly` engine on a **Windows 11 laptop with an
NVIDIA RTX 5060 Laptop (8 GB VRAM, Blackwell `sm_120`) and 32 GB RAM**.

Branch: `fork/add-inkling2-mini` (never push directly to `origin/main`).

---

## 1. Verified Model & GGUF Ground Truth

Extracted directly from the live GGUF headers of `unsloth/Inkling-2-Mini-GGUF` at pinned revision
`ea0c79b2ecd73669e774c8519b79bfabfddbc027` (`Mainline_Support` layout, matching merged `ggml-org/llama.cpp` PR `#32694`):

### 1.1 Pinned Files & Exact Sizes

| Local Filename | Remote Path (`unsloth/Inkling-2-Mini-GGUF`) | Bytes | Routed Experts (`arena_gb`) | Non-Expert Weights | Expert Types |
| --- | --- | ---: | ---: | ---: | --- |
| `Inkling-2-Mini-UD-Q4_K_M.gguf` | `Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file` | 16,610,544,608 | 14.78 GB (13.76 GiB) | 1.83 GB (1.71 GiB) | `NVFP4`, `Q5_K`, `Q8_0` |
| `Inkling-2-Mini-UD-Q4_K_XL.gguf` | `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` | 17,629,760,736 | 15.80 GB (14.71 GiB) | 1.83 GB (1.71 GiB) | `NVFP4`, `Q8_0` |
| `mmproj-BF16.gguf` | `mmproj-BF16.gguf` | 641,805,760 | — | 0.64 GB | `BF16` |

- Non-expert weight types in both quants: `Q8_0`, `BF16`, `F32`.
- **`UD-Q4_K_XL` advantage on Blackwell**: its routed experts are **pure `NVFP4` + `Q8_0`** (no `Q5_K`), which maps
  cleanly onto `strata-poly`'s native `NVFP4` paths (`src/poly/mmq_nvfp4.cu` and `src/kernels/cuda/native_mmvq.cu`).

### 1.2 Exact GGUF Metadata (`inkling2_loop.*`)

| Key | Value | Meaning |
| --- | --- | --- |
| `general.architecture` | `inkling2_loop` | Dispatched in `Engine::open()` (`src/poly/engine.cpp`) |
| `inkling2_loop.block_count` | `24` | Physical weight layers (`blk.0` .. `blk.23`) |
| `inkling2_loop.loop.start` / `end` / `count` | `6` / `18` / `3` | Core `blk.6..17` runs 3× → **48 logical layers/token** |
| `inkling2_loop.loop.gate_rank` | `64` | Low-rank loop controller dimension |
| `inkling2_loop.loop.varying` | `False` | Fixed 3 passes |
| `inkling2_loop.embedding_length` | `2304` | Hidden size |
| `inkling2_loop.vocab_size` / `unpadded` | `201024` / `200058` | Padded / unpadded vocabulary |
| `inkling2_loop.context_length` | `1048576` | 1M native context window |
| `inkling2_loop.attention.head_count` | `[24 × 24]` | 24 query heads |
| `inkling2_loop.attention.head_count_kv` | `[8 × 24]` | 8 KV heads (GQA 3:1) |
| `inkling2_loop.attention.key_length` / `value_length` | `128` / `128` | Global & SWA head dim = 128 |
| `inkling2_loop.attention.sliding_window` | `1024` | Local SWA window size |
| `inkling2_loop.attention.sliding_window_pattern` | 20 `True`, 4 `False` | Global layers at physical indices **`5, 11, 17, 23`** |
| `inkling2_loop.attention.relative_length` / `extent` | `16` / `1024` | Relative-position attention (`d_rel=16`, `rel_extent=1024`) |
| `inkling2_loop.attention.qk_clip` / `relative_clip` | `22.6274166` / `8.0` | QK and relative L2 clip thresholds |
| `inkling2_loop.attention.log_scale_floor` / `alpha` | `8192` / `0.1` | Attention log-scaling past 8K tokens |
| `inkling2_loop.attention.layer_norm_rms_epsilon` | `1e-6` | RMSNorm epsilon |
| `inkling2_loop.rope.dimension_count` | `0` | **No RoPE** (uses relative-position attention) |
| `inkling2_loop.shortconv.l_cache` | `4` | Causal short convolution kernel length |
| `inkling2_loop.attn_res.block_size` | `6` | Cross-layer Attention Residuals block size |
| `inkling2_loop.expert_count` / `used_count` | `128` / `8` | Top-8 of 128 routed experts (0 shared experts) |
| `inkling2_loop.expert_feed_forward_length` | `1152` | Per-expert FFN intermediate width |
| `inkling2_loop.expert_weights_scale` | `8.0` | Router weight scale |
| `inkling2_loop.activation.situ_beta` / `situ_linear_beta` | `4.0` / `25.0` | MoE gate/up softcap betas |
| `inkling2_loop.final_logit_softcapping` | `30.0` | Final logits softcap (`30 * tanh(x / 30)`) |
| `general.sampling.*` | `temp=1.0, top_p=1.0, top_k=0, min_p=0.0` | Recommended interactive sampling defaults |

### 1.3 Complete GGUF Tensor Inventory (744 Tensors)

- **Global / non-block tensors**:
  - `token_embd.weight` `[2304, 201024]` `Q8_0`
  - `token_embd_norm.weight` `[2304]` `F32`
  - `output.weight` `[2304, 201024]` `Q8_0`
  - `output_norm.weight` `[2304]` `F32`
  - `output_res_query.weight` `[2304]` `F32`
- **Every physical layer `blk.N` (`N = 0..23`)**:
  - Attention: `attn_norm.weight` `[2304]`, `attn_q.weight` `[2304, 3072]`, `attn_k.weight` `[2304, 1024]`,
    `attn_v.weight` `[2304, 1024]`, `attn_gate.weight` `[2304, 3072]`, `attn_output.weight` `[3072, 2304]`,
    `attn_rel.weight` `[2304, 384]`, `attn_rel_bias.weight` `[16, 1024]`, `attn_q_clip.weight` `[128, 24]`,
    `attn_rel_clip.weight` `[16, 24]`, `attn_sink_proj.weight` `[2304, 24]` (`BF16`),
    `attn_k_conv.weight` `[4, 1024]`, `attn_v_conv.weight` `[4, 1024]`, `attn_post_conv.weight` `[4, 2304]`.
  - Attention Residuals: `attn_res_query.weight` `[2304]` (layers `1..23`), `ffn_res_query.weight` `[2304]` (all layers).
  - MoE: `ffn_norm.weight` `[2304]`, `ffn_gate_inp.weight` `[2304, 128]`, `exp_probs_b.bias` `[128]`,
    **`ffn_gate_up_exps.weight` `[2304, 2304, 128]`** + **`ffn_gate_up_exps.scale` `[128]`**,
    **`ffn_down_exps.weight` `[1152, 2304, 128]`** + **`ffn_down_exps.scale` `[128]`**,
    `ffn_post_conv.weight` `[4, 2304]`.
- **Global layers only (`blk.5`, `blk.11`, `blk.17`, `blk.23`)**:
  - `attn_k_conv_proj.weight` `[2304, 32]` (`BF16`)
  - `attn_k_inp_norm.weight` `[2304]` (`F32`)
- **Looped core layers only (`blk.6` .. `blk.17`)**:
  - `loop_{q,k,v,ffn}_ctrl_down.weight` `[4608, 64]` (`F32`)
  - `loop_{q,k,v,ffn}_ctrl_forget.weight` `[64, 2304]` (`F32`)
  - `loop_{q,k,v,ffn}_ctrl_gain.weight` `[64, 2304]` (`F32`)
  - `loop_msg_query.weight` `[2304, 4]` (`F32`)

> **Critical bug to avoid (`ggml-org/llama.cpp#32704`)**: On passes 2 and 3 of the looped core (`blk.6..17`), each
> physical layer's per-expert NVFP4 global scales (`ffn_gate_up_exps.scale` and `ffn_down_exps.scale`) **must remain
> bound** on every pass. Dropping them on passes 2 & 3 causes perplexity to explode from ~7.65 to ~130.

---

## 2. KV Cache, Memory Budget & Realistic RTX 5060 Expectations

Unrolled 48 logical steps = **8 global** (`blk.5` ×1, `blk.11` ×3, `blk.17` ×3, `blk.23` ×1) + **40 local SWA** (`window = 1024`).

- Per K or V row (`n_kv × head_dim = 8 × 128 = 1,024` values):
  - **FP16**: `2,048 B/row` → `4,096 B/tok/layer`
  - **Q8_0** (32 int8 codes + 2 B fp16 scale per 32 values, see [Q8_KV.md](../Q8_KV.md)):
    `1,024 + 64 = 1,088 B/row` → `2,176 B/tok/layer` (**46.88% smaller**)
- **8 global layers**: `32,768 B/tok` (FP16) vs **`17,408 B/tok`** (Q8_0)
- **40 local SWA rings** (`1024 + 1024` prompt-batch rows): `320 MiB` (FP16) vs **`170 MiB`** (Q8_0)

| Context | FP16 KV Total | Q8_0 KV Total | Est. VRAM Left for Experts on 8 GB Card (Q8_0) | Notes |
| --- | ---: | ---: | ---: | --- |
| 32K | 1.31 GiB | 0.70 GiB | ~3.7 GiB (~25% of experts) | Benchmark first |
| 64K | 2.31 GiB | 1.23 GiB | ~3.2 GiB (~22% of experts) | Benchmark second |
| **128K (default)** | 4.31 GiB | **2.29 GiB** | **~2.1 GiB (~14% of experts)** | Keep as default only if measured decode speed holds up |
| 192K | 6.31 GiB | 3.35 GiB | ~1.0 GiB | Tight expert cache |
| 256K | 8.31 GiB | 4.42 GiB | ~0 GiB | Full CPU expert offload |

### Realistic Speed Expectations (Not Yet Measured)
Each decode token evaluates **384 experts** (48 logical steps × top-8 experts ≈ ~2 GB of expert weights). At 128K on
an 8 GB card, only ~2.1 GiB (~14% of the 14.78 GB experts) fits in VRAM, so most expert reads come from system RAM
(estimated ~10–20 tok/s decode once working, unmeasured). **Measure 32K and 64K first**, then 128K; keep 128K as
`setup.py`'s default only if measured speed at 128K is acceptable on the RTX 5060 Laptop.

---

## 3. Required Shared-Engine Changes (`src/poly/engine.cpp`)

Before implementing `Inkling2`'s forward kernels, two shared-engine items must be in place:

1. **Joined `ffn_gate_up_exps.scale` support** (**already applied** in `src/poly/engine.cpp:1622-1639`):
   - Previously `engine.cpp` threw `"NVFP4 scales of a joined gate and up are not supported"`.
   - It now uploads `ffn_gate_up_exps.scale` (F32 `[n_expert_]`) once and binds the same `Dev` pointer/host vector
     to both `L.s_gate`/`L.h_s_gate` and `L.s_up`/`L.h_s_up`. Models without `ffn_gate_up_exps.scale` take the
     unchanged `ffn_gate_exps.scale` / `ffn_up_exps.scale` path.

2. **Logical-to-physical layer schedule in `Engine::run()`** (`src/poly/engine.cpp:4969-5000`, **to implement in Phase 1**):
   - `Engine` currently loops `for (int il = lo_; il < hi_; ++il)` over `layers_`, where each `Layer` owns its
     VRAM expert cache (`L.slot`), NVFP4 scales (`s_gate`/`s_up`/`s_down`), and CUDA graph handles.
   - **Do NOT duplicate `Layer` 48 times** — that would triple the core layers' VRAM expert cache or risk dropping
     scales on passes 2 & 3.
   - **Design**:
     - Keep **24 physical `Layer` entries** (`layers_[0..23]`) so each physical layer has **one** VRAM expert-slot
       pool and **one** set of NVFP4 scales shared across all 3 passes.
     - Add two virtuals on `Engine` (defaulting to identity so every existing model is 100% untouched/byte-identical):
       - `virtual int n_steps() const { return hi_ - lo_; }`
       - `virtual int step_phys(int s) const { return lo_ + s; }`
     - In `Inkling2`, `n_steps()` returns `48` and `step_phys(s)` returns `steps_[s].phys` (`0..5`, `6..17` ×3, `18..23`).
     - In `Engine::run()`, loop `for (int s = 0; s < n_steps(); ++s)` with `int phys = step_phys(s); Layer& L = layers_[phys];`:
       - Pass `s` (logical step `0..47`) to `layer_head(L, s, n)` and `layer_tail(L, s, n)` so `Inkling2` indexes its
         own per-step KV cache (`kv_slots_[s]`) and loop pass (`steps_[s].pass`).
       - Pass `phys` (physical layer `0..23`) and `L` to `moe()`, `moe_card()`, and `moe_rows()` so all 3 core passes
         share the same physical expert cache and NVFP4 scales.
       - Key per-step CUDA graph caches (`heads`, `tails`) by logical step `s` (not `phys`) so step 6 (pass 0),
         step 18 (pass 1), and step 30 (pass 2) each record their own KV buffer pointers.

---

## 4. Phased Implementation & Optimization Plan

Follow strict order: **Reference Oracle → Exact Unfused Forward → Offload/NVFP4 → Kernel Fusion & Graphs → Vision → Compile/Regression Checks**. Never fuse or optimize before Stage 1 & Stage 2 parity pass.

### Phase 0 — Reference Oracle on RTX 5060 Laptop
1. Install Visual Studio Build Tools (C++), CUDA Toolkit (12.8 or 13.0), and CMake.
2. Pull this branch and download the model (`UD-Q4_K_XL` recommended first since its experts are pure `NVFP4`/`Q8_0`):
   ```powershell
   git fetch fork
   git checkout add-inkling2-mini
   python setup.py --family inkling2 --model UD-Q4_K_XL --vision no --yes --no-start
   ```
3. Build mainline `llama.cpp` pinned to PR `#32704` commit **`2a28df0f9bd55c4975b6bffb35229a6020720f13`**
   *(PR `#32704` is open as of 2026-11-21; pinning this commit keeps the reference reproducible)*:
   ```powershell
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   git fetch origin pull/32704/head:inkling2-loop-fix
   git checkout 2a28df0f9bd55c4975b6bffb35229a6020720f13
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j --target llama-cli llama-server llama-eval-callback
   ```
4. **Never run `-ngl 99` alone on an 8 GB VRAM card** — a 17.6 GB model will silently spill into Windows shared GPU
   memory and invalidate the baseline. Offload routed experts to CPU (`-ot "exps=CPU"` or tune `--n-cpu-moe N`):
   ```powershell
   # Greedy parity reference (--temp 0 --seed 42):
   .\build\bin\Release\llama-cli.exe `
     -m ..\models\inkling2-ud-q4_k_xl\Inkling-2-Mini-UD-Q4_K_XL.gguf `
     --jinja -ngl 99 -ot "exps=CPU" -c 32768 -ctk q8_0 -ctv q8_0 `
     --temp 0 --seed 42 -n 128 -p "Explain quicksort in three sentences."

   # Interactive / server baseline (recommended sampling):
   .\build\bin\Release\llama-server.exe `
     -m ..\models\inkling2-ud-q4_k_xl\Inkling-2-Mini-UD-Q4_K_XL.gguf `
     --jinja -ngl 99 -ot "exps=CPU" -c 65536 -ctk q8_0 -ctv q8_0 `
     --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --port 8001
   ```
5. Save greedy reference outputs on 5–10 fixed prompts (short factual, coding, and prefixes of a 3,740-token prompt)
   plus `llama-eval-callback` tensor dumps for Stage 1.

### Phase 1 — Exact Unfused Forward Pass (`src/poly/inkling2.cpp`)
- **Math reference**: Port every formula **directly from `ggml-org/llama.cpp`'s `src/models/inkling2.cpp`** at commit
  `2a28df0f9bd55c4975b6bffb35229a6020720f13` (relative-position attention, QK/rel L2 clip, log-scaling, value norm
  without gamma, shortconv `k=4`, `attn_k_conv_proj` on global layers, rank-64 loop gates, block-6 attention
  residuals, router + `exp_probs_b.bias`, situ softcaps `4.0`/`25.0`, and `final_logit_softcapping = 30.0`).
- **PolyStrata code templates**:
  - `src/poly/flint.cpp`: hybrid SWA ring buffers (`ring_ = std::min(opt_.max_context, window_ + rows_)` inside
    `alloc_caches()`, **never** `load_geometry()`), Q8_0 KV, `save_layer()` / `load_extra()`.
  - `src/poly/lfm2moe.cpp`: causal shortconv (`l_cache = 4`) state management.
  - `src/poly/engine.cpp`: logical/physical step schedule (Section 3.2) and `ffn_gate_up_exps.scale` (already done).
- **Keep every kernel simple and FP32-accumulated** in Phase 1; do not enable Blackwell FP4×FP4 activation
  quantization or kernel fusion until Stage 1 & Stage 2 pass.

### Phase 2 — Memory Placement & Offload Optimization
1. **Enable `--kv q8_0`**: Verify 2.29 GiB total KV at 128K (vs 4.31 GiB FP16).
2. **Loop-core-weighted VRAM LRU expert cache**: Weight hits on core layers `6..17` by `loop_count_` (**3×**) so
   hot core experts stay pinned in VRAM over prelude/epilogue experts.
3. **Native NVFP4 execution on Blackwell (`sm_120`)**: Enable `src/poly/mmq_nvfp4.cu` FP4×FP4 prefill and `iq_mmvq`
   (`src/kernels/cuda/native_mmvq.cu`) decode, then re-verify Stage 2 greedy parity.
4. **Profile with Nsight Systems (`nsys`)**: Capture 10 decode steps + 1 prefill batch at 32K, 64K, and 128K before
   writing fused kernels.

### Phase 3 — Kernel Fusion & CUDA Graphs
Apply one fusion at a time, re-running Stage 2 greedy parity after **each** fusion:
1. **Attention prep fusion**: Fuse `shortconv (k=4)` + `QK/rel L2 clip` + `value norm` + `Q8_0 KV quantize-and-store`.
2. **Flash-decode fusion**: Fold `relative-position bias (d_rel=16)` + `log-scaling` + `learned sink` into a single
   decode attention kernel (1 VRAM pass over K/V).
3. **Residual & loop-gate fusion**: Fuse `rank-64 loop gate` + `attention residual (block 6)` + `post-conv` + `RMSNorm`.
4. **MoE activation fusion**: Fuse `situ softcap (4.0/25.0)` + activation + input quantization for `ffn_down_exps` in
   registers.
5. **CUDA Graph capture**: Set `L.fixed = true` so `Engine::passes_` / `tail_` replays all 48 logical steps with
   minimal CPU launch overhead.

### Phase 4 — Vision & Deferred Follow-Ups
- **Vision**: Verify `mmproj-BF16.gguf` through `strata-vision` on image prompts.
- **DSpark speculative decoding (`dspark-Inkling-2-Mini-Q8_0.gguf`, 0.4 GB)**: **Deferred to a separate follow-up
  branch/PR** — the drafter file is not pinned yet and its 3-layer Markov/RNN-hidden architecture (`mtp_config`) has
  no existing `draft_step()` path in `Engine`.

### Phase 5 — Cross-Backend Compile Check & Regression Hygiene
- **HIP**: Compile-only check on the Windows NVIDIA laptop if ROCm/HIP SDK is installed (runtime HIP validation
  requires AMD hardware or CI; document what was and was not tested per `AGENTS.md`).
- Confirm all existing models (`qwen35moe`/Groot, `flint`, `gemma4`, `lfm2moe`) remain byte-identical on default paths.
- Record measured RTX 5060 Laptop numbers (32K, 64K, 128K) in `docs/models/inkling2.md` and `bench/results/`.

---

## 5. 4-Stage Verification & Test Plan (Run on RTX 5060 Laptop)

### Stage 1 — Layer-by-Layer Tensor Parity (Unfused, Non-FP4×FP4 Path)
- **Requirement**: Run **without** Blackwell FP4×FP4 activation quantization (activation quantization introduces noise
  that will falsely fail a 0.9999 cosine threshold).
- **How to dump hidden states**:
  - `llama.cpp` side: run `llama-eval-callback` (built in Phase 0) on a 1-token and 16-token prompt to dump tensors
    after `blk.0`, `blk.5` (1st global), `blk.6` **pass 1, pass 2, pass 3**, `blk.23`, and final logits.
  - `strata-poly` side: add a temporary opt-in `STRATA_INKLING2_DUMP=1` `cudaMemcpy` hook at the end of
    `Inkling2::layer_tail()` (removed before PR) dumping the same checkpoints to `.bin` files.
- **Pass criteria**: Cosine similarity `>= 0.9999` at every checkpoint; zero NaN/Inf; pass 2 & pass 3 match pass 1
  accuracy (confirms per-expert NVFP4 scales stay bound).

### Stage 2 — Greedy Token-by-Token Parity
- Run with `temperature = 0`, `--no-draft` on both `UD-Q4_K_M` and `UD-Q4_K_XL` against `llama.cpp` (`2a28df0f9b`):
  - 5 short factual/coding prompts + prefixes of a 3,740-token prompt (100, 600, 1500, 3740 tokens — same method as
    Flint commit `5ff592f1`).
- **Pass criteria**: Exact token match on short and long prompts, **or** any first divergence occurs only where
  `llama.cpp`'s top-2 logits are within ~0.5 log-prob (verify by printing top-3 logits at the diverging step).

### Stage 3 — Q8_0 KV & Long-Context Needle Verification
```powershell
# 1. Teacher-forced FP16 vs Q8_0 KV agreement:
python tools/kv_agree.py `
  --model models/inkling2-ud-q4_k_xl/Inkling-2-Mini-UD-Q4_K_XL.gguf `
  --bin build/Release/strata-poly.exe `
  --ref fp16 --kv q8_0 --prompts 5 --max-new 128 --max-context 2048 --min-agree 0.97

# 2. Needle recall at 32K, 64K, and 128K (requires running PolyStrata server on :8080):
python tools/needle_bench.py `
  --url http://127.0.0.1:8080 `
  --lengths 32k,64k,128k --depths 10,50,90 `
  --out bench/inkling2_needle.json
```
- **Pass criteria**: FP16/Q8_0 agreement `>= 97%`; 100% needle recall at 32K, 64K, and 128K (confirms SWA ring
  wraparound past 1,024 tokens and log-scaling past 8,192 tokens).

### Stage 4 — Same-Day A/B Benchmark & Regression Suite
```powershell
# 1. Generate prompts and run same-day A/B comparison against llama.cpp (measure 32K, 64K, then 128K):
python bench/bench_vs_llama.py prompts --out bench/inkling2_vs_llama --tokenizer models/inkling2-ud-q4_k_xl
python bench/bench_vs_llama.py run --out bench/inkling2_vs_llama --machine rtx5060-laptop --configs bench/inkling2_configs.json --rounds 1
python bench/bench_vs_llama.py summary --out bench/inkling2_vs_llama

# 2. Offline setup, server, and MCP unit tests:
$env:PYTHONPATH="."
python tools/test_setup_inkling2.py
python tools/test_setup_groot.py
python tools/test_setup_menu.py
python tools/test_setup_config.py
python tools/test_setup_pins.py
python serve/test_models.py
python tools/test_strata_mcp.py
```
- **Pass criteria**: Peak VRAM `<= 7.5 GiB`, peak RAM `<= 26 GiB`; all unit tests pass; existing `groot`
  (`qwen35moe`) greedy output unchanged.

---

## 6. GPU Portability Notes (One GPU vs All GPUs)

- **~80–90% of speedups transfer automatically to all GPUs (NVIDIA RTX 20–50 & AMD HIP)**:
  - Q8_0 KV cache, SWA ring buffers, 3× loop-core-weighted VRAM LRU expert cache, kernel fusion (fewer VRAM
    round-trips and launches across 48 logical layers), and CUDA/HIP Graph capture.
- **~10–20% is generation/vendor specific**:
  - Native **FP4×FP4** tensor-core prefill (`src/poly/mmq_nvfp4.cu`) is Blackwell (`sm_120`, RTX 50) specific;
    RTX 20/30/40 fall back to their existing MMA/`dp4a` paths automatically.
  - AMD HIP wavefront (32 vs 64 lanes) and dot-product intrinsics use thin `#if defined(__HIPCC__)` guards inside
    shared kernel headers.
