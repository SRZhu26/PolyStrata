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
| `inkling2_loop.vocab_size` / `unpadded` | `201024` / `200058` | **Only the first `200,058` rows are scored**; mask the 966 padding rows before softcap and sampling |
| `inkling2_loop.context_length` | `1048576` | 1M native context window |
| `inkling2_loop.attention.head_count` | `[24 × 24]` | 24 query heads |
| `inkling2_loop.attention.head_count_kv` | `[8 × 24]` | 8 KV heads (GQA 3:1) |
| `inkling2_loop.attention.key_length` / `value_length` | `128` / `128` | Global & SWA head dim = 128 |
| `inkling2_loop.attention.sliding_window` | `1024` | Local SWA window size |
| `inkling2_loop.attention.sliding_window_pattern` | 20 `True`, 4 `False` | Global layers at physical indices **`5, 11, 17, 23`** |
| `inkling2_loop.attention.relative_length` / `extent` | `16` / `1024` | Relative-position attention (`d_rel=16`, `rel_extent=1024`) |
| `inkling2_loop.attention.qk_clip` / `relative_clip` | `22.6274166` / `8.0` | QK and relative L2 clip thresholds |
| `inkling2_loop.attention.log_scale_floor` / `alpha` | `8192` / `0.1` | Attention log-scaling past 8K tokens (**global layers `5, 11, 17, 23` only**) |
| `inkling2_loop.attention.layer_norm_rms_epsilon` | `1e-6` | RMSNorm epsilon |
| `inkling2_loop.rope.dimension_count` | `0` | **No RoPE** (uses relative-position attention) |
| `inkling2_loop.shortconv.l_cache` | `4` | Causal short convolution kernel length |
| `inkling2_loop.attn_res.block_size` | `6` | Cross-layer Attention Residuals block size |
| `inkling2_loop.expert_count` / `used_count` | `128` / `8` | Top-8 of 128 routed experts (0 shared experts) |
| `inkling2_loop.expert_feed_forward_length` | `1152` | Per-expert FFN intermediate width |
| `inkling2_loop.expert_weights_scale` | `8.0` | Router weight scale |
| `inkling2_loop.activation.situ_beta` / `situ_linear_beta` | `4.0` / `25.0` | MoE gate/up softcap betas |
| `inkling2_loop.final_logit_softcapping` | `30.0` | Final logits softcap (`30 * tanh(x / 30)`), after masking 966 padding rows |
| `general.sampling.*` | `temp=1.0, top_p=1.0, top_k=0, min_p=0.0` | Recommended interactive sampling defaults |

### 1.3 Complete GGUF Tensor Inventory

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
- **Optional / global-layer tensors (`blk.5`, `blk.11`, `blk.17`, `blk.23` — check against the file by name)**:
  - `attn_k_conv_proj.weight` `[2304, 32]` (`BF16`) and `attn_v_conv_proj.weight` (per-head conv taps for K **and V** — verify presence at load)
  - `attn_k_inp_norm.weight` `[2304]` (`F32`), plus optional `attn_q_inp_norm.weight` and `attn_v_inp_norm.weight`
  - **Loader rule**: look up optional tensors by name and throw on any unconsumed `blk.N.*` tensor so a missing
    feature fails loudly instead of silently producing wrong output.
- **Looped core layers only (`blk.6` .. `blk.17`)**:
  - `loop_{q,k,v,ffn}_ctrl_down.weight` `[4608, 64]` (`F32`)
  - `loop_{q,k,v,ffn}_ctrl_forget.weight` `[64, 2304]` (`F32`)
  - `loop_{q,k,v,ffn}_ctrl_gain.weight` `[64, 2304]` (`F32`)
  - `loop_msg_query.weight` `[2304, 4]` (`F32`)

> **Critical bug to avoid (`ggml-org/llama.cpp#32704`)**: On passes 2 and 3 of the looped core (`blk.6..17`), each
> physical layer's per-expert NVFP4 global scales (`ffn_gate_up_exps.scale` and `ffn_down_exps.scale`) **must remain
> bound** on every pass. Dropping them on passes 2 & 3 causes perplexity to explode from ~7.65 to ~130.

---

## 2. KV Cache, Memory Budget & Context Default

Unrolled 48 logical steps = **8 global** (`blk.5` ×1, `blk.11` ×3, `blk.17` ×3, `blk.23` ×1) + **40 local SWA** (`window = 1024`).

- Per K or V row (`n_kv × head_dim = 8 × 128 = 1,024` values):
  - **FP16**: `2,048 B/row` → `4,096 B/tok/layer`
  - **Q8_0** (32 int8 codes + 2 B fp16 scale per 32 values, see [Q8_KV.md](../Q8_KV.md)):
    `1,024 + 64 = 1,088 B/row` → `2,176 B/tok/layer` (**46.88% smaller**)
- **8 global layers**: `32,768 B/tok` (FP16) vs **`17,408 B/tok`** (Q8_0)
- **40 local SWA rings** (`1024 + 1024` prompt-batch rows): `320 MiB` (FP16) vs **`170 MiB`** (Q8_0)

| Context | FP16 KV Total | Q8_0 KV Total | Est. VRAM Left for Experts on 8 GB Card (Q8_0) | Notes |
| --- | ---: | ---: | ---: | --- |
| **32K (default)** | 1.31 GiB | **0.70 GiB** | **~3.7 GiB (~25% of experts)** | Default until 128K is measured |
| 64K | 2.31 GiB | 1.23 GiB | ~3.2 GiB (~22% of experts) | Option via `--context 65536` |
| 128K | 4.31 GiB | 2.29 GiB | ~2.1 GiB (~14% of experts) | Option via `--context 131072`; promote to default only after Stages 3 & 4 pass at 128K (`AGENTS.md`) |
| 192K | 6.31 GiB | 3.35 GiB | ~1.0 GiB | Tight expert cache |
| 256K | 8.31 GiB | 4.42 GiB | ~0 GiB | Full CPU expert offload |

---

## 3. Shared-Engine Architecture (`src/poly/engine.hpp`, `src/poly/engine.cpp`)

### 3.1 Joined `ffn_gate_up_exps.scale` Support (Already Applied)
- `src/poly/engine.cpp:1622-1639`: uploads `ffn_gate_up_exps.scale` (F32 `[n_expert_]`) once and binds the same
  `Dev` handle / host vector to both `L.s_gate`/`L.h_s_gate` and `L.s_up`/`L.h_s_up`. Models without
  `ffn_gate_up_exps.scale` take the unchanged path.
- `Inkling2::load_layer()` additionally checks non-NVFP4 expert layers using the **weight tensor's own type**
  (`tw->type`, since `load_layer()` runs before `load_moe()` sets `L.gu_type`/`L.d_type`), validates `F32`, calls
  `home(*t).payload(*t)`, and throws if any `.scale` element != `1.f` (kept inside `Inkling2` so Gemma 4's own
  non-NVFP4 `ffn_down_exps.scale` is untouched).

### 3.2 Logical-to-Physical Step Tracking in `Engine` (Already Applied)
- **24 physical `Layer` entries** (`layers_[0..23]`) share one VRAM expert-slot pool (`L.slot`) and one set of
  NVFP4 scales per physical layer across all 3 core passes — never duplicated 48×.
- `Engine` tracks the active visit via `int cur_step_ = 0` and five virtual hooks (defaulting to identity so every
  other model is 100% unchanged):
  - `n_steps()` (`48` for Inkling, `hi_ - lo_` by default)
  - `step_phys(s)` (`steps_[s].phys` for Inkling, `lo_ + s` by default)
  - `next_moe(il)` (`steps_[cur_step_ + 1].phys` for Inkling)
  - `halves_ok()` (`false` for Inkling until Phase 3, `true` by default)
  - `read_layer_major()` (`false` for Inkling until per-call state can be moved to host memory, `true` by default)
- All six engine loops updated in `src/poly/engine.cpp`:
  1. Whole-pass check, `head_key` key loop, whole-pass CUDA Graph recording, and stats (`~4971-5005`).
  2. Plain per-layer loop (`~5006-5016`).
  3. `Engine::step()` (`~5102`).
  4. `moe()` prefetch (`~4058`, via `next_moe(il)`).
  5. Two-halves window path (`~4889`, gated by `&& halves_ok()`).
  6. Long-prompt `Engine::read()` (`~5553-5590`, plus `read_rows_` opt-out at `~3070` via `read_layer_major()`).
     `Inkling2::save_layer()` must save the current visit (`cur_step_`): its SWA ring and its conv history.
- **CUDA graph safety across the 3 core passes**:
  - **Phase 1**: keep **`L.fixed = false`** in `Inkling2::load_layer()` (`L.tail`, `L.shared`, and `L.card_moe` are
    keyed by `rec` only, so enabling CUDA graphs before Phase 3 would replay pass 1's tail on passes 2 and 3).
  - **Phase 3**: add a pass-aware `tail_key()` virtual (defaulting to `0`, used as `rec + tail_key() * recs`) and
    combine `pass` with `room_key(n)` on global layers as `pass * (max_room_key + 1) + room_key(n)` so keys never
    collide.
- `Engine::layer()` (`~5401`) and `card_moe()` slot table (`~4266`, keyed by `il * n_expert_`) are intentionally
  unchanged: all 3 passes share the physical expert pool.

### 3.3 Per-Token State Across Visits
- **Within one call (scratch, sized by `rows_`):**
  - Depth-attention (Attention Residuals) history: up to **8 block states** of `n_embd` floats per token (`block_size = 6`).
  - Previous pass's `q`/`k`/`v`/`ffn` loop messages for core layers `6..17`, read by passes 2 and 3.
    *(Verify the exact tensor size from `src/models/inkling2.cpp` before sizing scratch — the controllers are rank 64,
    so the real buffer may be smaller than `12 × 4 × n_embd` per token.)*
- **Across calls (recurrent state, `Inkling2::recurrent_floats()`):**
  - Each of the 48 visits keeps its own causal-conv history (`l_cache - 1 = 3` rows) for K (`1024`), V (`1024`),
    attention post-conv (`2304`), and FFN post-conv (`2304`): `48 × 3 × 6,656 = 958,464` floats (~3.66 MiB).
  - Because it lives in `Engine`'s recurrent state, it must be saved and restored cleanly across prefix reuse,
    pinned sessions (`--pinned-session`), and `save_layer()` / `load_extra()`.
- **Phase 1 limits**: `halves_ok() = false`, `read_layer_major() = false`, `L.fixed = false`, and `--layer-split`
  refused in `load_geometry()`.

---

## 4. Phased Implementation & Optimization Plan

Order to execute: **Phase 0 (Oracle) → Phase 1 (Exact Unfused Forward) → Phase 1b (Tokenizer/Server) → Phase 2 (Offload/NVFP4) → Phase 3 (Fusion/Graphs) → Phase 4 (Vision) → Phase 5 (Compile/Regression)**.

### Phase 0 — Reference Oracle on RTX 5060 Laptop
1. Install Visual Studio Build Tools (C++), CUDA Toolkit (12.8 or 13.0), and CMake.
2. Checkout this branch and download the model (`UD-Q4_K_XL` recommended first: pure `NVFP4`/`Q8_0` experts):
   ```powershell
   git fetch fork
   git checkout add-inkling2-mini
   python setup.py --family inkling2 --model UD-Q4_K_XL --vision no --yes --no-start
   ```
3. Build mainline `llama.cpp` pinned to PR `#32704` commit **`2a28df0f9bd55c4975b6bffb35229a6020720f13`**
   *(PR `#32704` is open as of 2026-11-21; if `2a28df0f9b` does not load the file, use Unsloth's prebuilt
   `b12588-mix-c0c3de8`, the build ninfer-all measured against)*:
   ```powershell
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   git fetch origin pull/32704/head:inkling2-loop-fix
   git checkout 2a28df0f9bd55c4975b6bffb35229a6020720f13
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j --target llama-cli llama-bench llama-server llama-tokenize
   ```
4. First confirm the file loads, then measure the speed baseline using **`llama.cpp`'s own memory fitting**
   (`--fit-target`, no `-ngl`/`-ot` — ninfer-all measured `-ot "exps=CPU"` 2–8× slower):
   ```powershell
   # 0. Confirm the pinned build loads the Mainline_Support file:
   .\build\bin\Release\llama-cli.exe -m ..\models\inkling2-ud-q4_k_xl\Inkling-2-Mini-UD-Q4_K_XL.gguf -n 1 -p "hi"

   # 1. Speed baseline (llama.cpp fits the model to 8 GB itself):
   .\build\bin\Release\llama-bench.exe -m ..\models\inkling2-ud-q4_k_xl\Inkling-2-Mini-UD-Q4_K_XL.gguf `
     -fa 1 -ctk q8_0 -ctv q8_0 --fit-target 1024 --fit-ctx 32768 -p 2048 -n 128 -r 3

   # 2. Greedy reference (same fitting, --temp 0):
   .\build\bin\Release\llama-cli.exe -m ..\models\inkling2-ud-q4_k_xl\Inkling-2-Mini-UD-Q4_K_XL.gguf `
     --jinja -c 32768 -ctk q8_0 -ctv q8_0 --temp 0 --seed 42 -n 128 -p "Explain quicksort in three sentences."
   ```

### Phase 1 — Exact Unfused Forward Pass (`src/poly/inkling2.cpp`)
- **Math reference**: Port every formula **directly from `ggml-org/llama.cpp`'s `src/models/inkling2.cpp`** at commit
  `2a28df0f9bd55c4975b6bffb35229a6020720f13` (relative-position attention, QK/rel L2 clip, global-only log-scaling,
  value norm without gamma, shortconv `k=4`, global K/V conv taps, rank-64 loop gates, block-6 attention residuals,
  router + `exp_probs_b.bias`, situ softcaps `4.0`/`25.0`, 966-row vocab padding mask, `final_logit_softcapping = 30.0`).
- **PolyStrata code templates**:
  - `src/poly/flint.cpp`: hybrid SWA ring buffers (`ring_ = std::min(opt_.max_context, window_ + rows_)` inside
    `alloc_caches()`, **never** `load_geometry()`), Q8_0 KV, `save_layer()` / `load_extra()`.
  - `src/poly/lfm2moe.cpp`: causal shortconv (`l_cache = 4`) recurrent state.
- **Keep every kernel simple and FP32-accumulated** in Phase 1; do not enable Blackwell FP4×FP4 activation
  quantization or kernel fusion until Stage 1 & Stage 2 pass.

### Phase 1b — Tokenizer, Chat Template & Server (Required Before Stages 3 & 4)
1. Read `tokenizer.ggml.pre` from the GGUF. If its pre-tokenizer pattern is not already in `strata_tokenizer.py`,
   add it to `PRE_PATTERNS`.
2. Encode 20 mixed strings (code, CJK, emoji, numbers) and compare token IDs byte-for-byte against `llama-tokenize`.
3. Server (`serve/`): wire the GGUF chat template, expose `<|content_thinking|>` blocks as `reasoning_content`, and
   parse `<|content_invoke_tool_json|>` blocks into OpenAI `tool_calls`. Add unit tests in `serve/test_models.py`.

### Phase 2 — Memory Placement & Offload Optimization
1. **Enable `--kv q8_0`**: Verify 0.70 GiB total KV at 32K (2.29 GiB at 128K).
2. **Core expert cache accounting (no extra 3× multiplier)**: Check that `use_[phys * n_expert_ + e]` increments
   once per visit, so core experts (`blk.6..17`) are already counted 3× naturally — add **no** extra weighting.
   Optional: seed the cache from a recorded routing profile (ninfer-all's profile raised the 8 GB hit rate to 76.7%).
3. **Native NVFP4 execution on Blackwell (`sm_120`)**: Enable `src/poly/mmq_nvfp4.cu` FP4×FP4 prefill and `iq_mmvq`
   (`src/kernels/cuda/native_mmvq.cu`) decode, then re-verify Stage 1 NLL and Stage 2 greedy parity.
4. **Profile with Nsight Systems (`nsys`)**: Capture 10 decode steps + 1 prefill batch at 32K (then 64K/128K) before
   fusing kernels.

### Phase 3 — Kernel Fusion & CUDA Graphs
Apply one fusion at a time, re-running Stage 2 greedy parity after **each** change:
1. **Attention prep fusion**: Fuse `shortconv (k=4)` + `QK/rel L2 clip` + `value norm` + `Q8_0 KV quantize-and-store`.
2. **Flash-decode fusion**: Fold `relative-position bias (d_rel=16)` + global `log-scaling` + `learned sink` into a
   single decode attention kernel (1 VRAM pass over K/V).
3. **Residual & loop-gate fusion**: Fuse `rank-64 loop gate` + `attention residual (block 6)` + `post-conv` + `RMSNorm`.
4. **MoE activation fusion**: Fuse `situ softcap (4.0/25.0)` + activation + input quantization for `ffn_down_exps` in
   registers.
5. **CUDA Graph capture**: Set `L.fixed = true` (with `head_key` encoding `pass` + ring/window state) so
   `Engine::passes_` / `tail_` replays all 48 logical steps with minimal CPU launch overhead.

### Phase 4 — Vision & Deferred Follow-Ups
- **Vision**: Verify `mmproj-BF16.gguf` through `strata-vision` on image prompts.
- **DSpark speculative decoding (`dspark-Inkling-2-Mini-Q8_0.gguf`, 0.4 GB)**: **Deferred to a separate follow-up
  branch/PR** — the drafter file is not pinned yet and its 3-layer Markov/RNN-hidden architecture (`mtp_config`) has
  no existing `draft_step()` path in `Engine`.

### Phase 5 — Cross-Backend Compile Check & Regression Hygiene
Because `src/poly/engine.hpp` and `src/poly/engine.cpp` touch shared code, before any PR:
- Build CUDA on the laptop and compile HIP if ROCm/HIP SDK is available (runtime HIP validation requires AMD
  hardware or CI; document what was and was not tested per `AGENTS.md`).
- **Existing-model regression check**: greedy-compare one existing model (Groot `qwen35moe`, or Flint with
  `--kv q8_0`) against a `main` build on **both** a short prompt and a long prompt (`>= max_rows()`) that goes
  through `Engine::read()` to prove byte-identical default paths.
- Record measured RTX 5060 Laptop numbers (32K, 64K, 128K) in `docs/models/inkling2.md` and `bench/results/`.

---

## 5. 4-Stage Verification & Test Plan (Run on RTX 5060 Laptop)

### Stage 1 — FP64 Reference Parity & NLL Check (Unfused, Non-FP4×FP4 Path)
- **Reference**: Use ninfer-all's Apache-2.0 FP64 reference (`tools/reference/inkling2.py`; check its CLI flags first),
  which runs the model in FP64 over the exact decoded GGUF blocks.
- **Per-visit check** (add a temporary `STRATA_INKLING2_DUMP=1` `cudaMemcpy` hook at the end of `Inkling2::layer_tail()`):
  - Cosine similarity **`>= 0.999`** vs FP64 after `blk.0`, `blk.5`, `blk.6` on **passes 1, 2, and 3**, `blk.23`, and
    the final logits.
  - Zero NaN/Inf; passes 2 and 3 no worse than pass 1 (confirms per-expert NVFP4 scales stay bound).
- **NLL / perplexity check**:
  - Mean NLL within **0.1%** of FP64 on ~1,150 WikiText tokens (past the 1,024 SWA window).
  - Within **0.1%** of `llama.cpp`'s perplexity at `--context 4096`.
  - One run past **8,192 tokens** to exercise global-layer log-scaling.
  - *(Calibration reference: ninfer-all measured `+0.03%` vs FP64 and `-0.06%` vs `llama.cpp`.)*

### Stage 2 — Greedy Token-by-Token Parity vs `llama.cpp`
- Run with `temperature = 0`, `--no-draft` on both `UD-Q4_K_M` and `UD-Q4_K_XL` against `llama.cpp` (`2a28df0f9b` or
  `b12588-mix-c0c3de8`):
  - 5 short factual/coding prompts + prefixes of a 3,740-token prompt (100, 600, 1500, 3740 tokens — same method as
    Flint commit `5ff592f1`).
- **Pass criteria**: Exact token match, **or** any first divergence occurs only where `llama.cpp`'s top-2 logits are
  within ~0.5 log-prob (verify by printing top-3 logits at the diverging step).

### Stage 3 — Q8_0 KV & Long-Context Needle Verification (Requires Phase 1b)
```powershell
# 1. Teacher-forced FP16 vs Q8_0 KV agreement (--bin is the engine path setup prints at the end):
python tools/kv_agree.py `
  --model models/inkling2-ud-q4_k_xl/Inkling-2-Mini-UD-Q4_K_XL.gguf `
  --bin <engine dir>\strata-poly.exe `
  --ref fp16 --kv q8_0 --prompts 5 --max-new 128 --max-context 2048 --min-agree 0.97

# 2. Needle recall at 32K, 64K, and 128K (requires running PolyStrata server on :8080):
python tools/needle_bench.py `
  --url http://127.0.0.1:8080 `
  --lengths 32k,64k,128k --depths 10,50,90 `
  --out bench/inkling2_needle.json
```
- **Pass criteria**: FP16/Q8_0 agreement `>= 97%`; 100% needle recall at 32K, 64K, and 128K.

### Stage 4 — Same-Day A/B Benchmark & Regression Suite (Requires Phase 1b)
Create `bench/inkling2_machine.json` (verify exact JSON field names like `"-fa","on"` against `bench/bench_vs_llama.py`
before running):
```json
{
  "gguf": "models\\inkling2-ud-q4_k_xl\\Inkling-2-Mini-UD-Q4_K_XL.gguf",
  "strata_cfg": "strata-inkling2-ud-q4_k_xl.json",
  "llama_server": "..\\llama.cpp\\build\\bin\\Release\\llama-server.exe",
  "configs": [
    {"name": "llama", "kind": "llama", "args": ["--fit-target", "1024", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"]},
    {"name": "strata", "kind": "strata", "args": []}
  ]
}
```
```powershell
# 1. Same-day A/B benchmark (bench_vs_llama.py has 4K and 32K cells; measure 64K/128K via needle_bench or added cells):
python bench/bench_vs_llama.py prompts --out bench/inkling2_vs_llama --repo . `
  --tokenizer <pack>\tokenizer --corpus <text file of 40K+ tokens>
python bench/bench_vs_llama.py run --machine bench/inkling2_machine.json --out bench/inkling2_vs_llama `
  --configs llama,strata --rounds 3
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
- **Pass criteria**:
  - At **32K with Q8_0 KV**: decode **`>= 1.5×`** the same-day `llama.cpp` baseline from Phase 0.
    *(Reference point: ninfer-all on a desktop RTX 5060 8 GB decoded 33.1 tok/s and prefilled 1,267 tok/s at 2K
    against `llama.cpp`'s 14.7 and 305; expect less on the laptop.)*
  - Peak VRAM `<= 7.5 GiB`, peak RAM `<= 26 GiB`.
  - All unit tests pass; existing `groot` (`qwen35moe`) greedy output unchanged.

---

## 6. GPU Portability Notes (One GPU vs All GPUs)

- **~80–90% of speedups transfer automatically to all GPUs (NVIDIA RTX 20–50 & AMD HIP)**:
  - Q8_0 KV cache, SWA ring buffers, VRAM LRU expert cache (naturally counting core experts 3× per token), kernel
    fusion, and CUDA/HIP Graph capture.
- **~10–20% is generation/vendor specific**:
  - Native **FP4×FP4** tensor-core prefill (`src/poly/mmq_nvfp4.cu`) is Blackwell (`sm_120`, RTX 50) specific;
    RTX 20/30/40 fall back to their existing MMA/`dp4a` paths automatically.
  - AMD HIP wavefront (32 vs 64 lanes) and dot-product intrinsics use thin `#if defined(__HIPCC__)` guards inside
    shared kernel headers.
