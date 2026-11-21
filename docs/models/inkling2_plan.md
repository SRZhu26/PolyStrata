# Inkling 2 Mini 26B-A5B (`inkling2_loop`) — Implementation, Optimization & Test Plan

Self-contained engineering handoff for bringing up, optimizing, and verifying **Thinking Machines Lab's Inkling 2 Mini**
(`general.architecture = "inkling2_loop"`) in PolyStrata's `strata-poly` engine on a **Windows 11 laptop with an
NVIDIA RTX 5060 Laptop (8 GB VRAM, Blackwell `sm_120`) and 32 GB RAM**.

Branch: `fork/add-inkling2-mini` (never push directly to `origin/main`).

---

## 1. Verified Model & GGUF Ground Truth

Extracted directly from the live GGUF headers of `unsloth/Inkling-2-Mini-GGUF` at pinned revision
`ea0c79b2ecd73669e774c8519b79bfabfddbc027` (`Mainline_Support` layout, matching `ggml-org/llama.cpp` PR `#32694`):

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
| `general.sampling.*` | `temp=1.0, top_p=1.0, top_k=0, min_p=0.0` | Recommended sampling defaults |

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

## 2. KV Cache & 32 GB RAM / 8 GB VRAM Sizing

Unrolled 48 logical steps = **8 global** (`blk.5` ×1, `blk.11` ×3, `blk.17` ×3, `blk.23` ×1) + **40 local SWA** (`window = 1024`).

- Per K or V row (`n_kv × head_dim = 8 × 128 = 1,024` values):
  - **FP16**: `2,048 B/row` → `4,096 B/tok/layer`
  - **Q8_0** (32 int8 codes + 2 B fp16 scale per 32 values, see [Q8_KV.md](../Q8_KV.md)):
    `1,024 + 64 = 1,088 B/row` → `2,176 B/tok/layer` (**46.88% smaller**)
- **8 global layers**: `32,768 B/tok` (FP16) vs **`17,408 B/tok`** (Q8_0)
- **40 local SWA rings** (`1024 + 1024` prompt-batch rows): `320 MiB` (FP16) vs **`170 MiB`** (Q8_0)

| Context | FP16 KV Total | Q8_0 KV Total | Est. VRAM Left for Experts on 8 GB Card (Q8_0) | Notes |
| --- | ---: | ---: | ---: | --- |
| 32K | 1.31 GiB | 0.70 GiB | ~3.7 GiB | Fastest decode |
| 64K | 2.31 GiB | 1.23 GiB | ~3.2 GiB | Great speed + context balance |
| **128K (default)** | 4.31 GiB | **2.29 GiB** | **~2.1 GiB** | **Default for 32 GB RAM / 8 GB VRAM** |
| 192K | 6.31 GiB | 3.35 GiB | ~1.0 GiB | Tight expert cache |
| 256K | 8.31 GiB | 4.42 GiB | ~0 GiB | Full CPU expert offload |

---

## 3. Phased Implementation & Optimization Plan

Follow strict order: **Reference Oracle → Exact Unfused Forward → Offload/NVFP4 → Kernel Fusion & Graphs → DSpark/Vision → HIP/CUDA Parity**. Never fuse or optimize before Stage 1 & Stage 2 parity pass.

### Phase 0 — Baseline Oracle on RTX 5060 Laptop
1. Install Visual Studio Build Tools (C++), CUDA Toolkit (12.8 or 13.0), and CMake.
2. Pull this branch and download the model (`UD-Q4_K_M` or `UD-Q4_K_XL`):
   ```powershell
   git fetch fork
   git checkout add-inkling2-mini
   python setup.py --family inkling2 --model UD-Q4_K_XL --vision no --yes --no-start
   ```
3. Build mainline `llama.cpp` with PR `#32704`:
   ```powershell
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   git fetch origin pull/32704/head:inkling2-loop-fix
   git checkout inkling2-loop-fix
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j --target llama-cli llama-server
   ```
4. Record baseline speed/VRAM/RAM at 128K + `-ctk q8_0 -ctv q8_0` and save greedy (`--temp 0`) outputs on 5–10 fixed prompts (short factual, coding, and prefixes of a 3,740-token prompt) as the exact parity oracle.

### Phase 1 — Exact Unfused Forward Pass (`src/poly/inkling2.cpp`)
Reference templates already in this repo:
- `src/poly/flint.cpp`: hybrid SWA ring buffers + global KV + Q8_0 KV + `save_layer()` / `load_extra()`.
- `src/poly/lfm2moe.cpp`: causal shortconv (`l_cache = 4`) state handling.
- `src/poly/mmq_nvfp4.cu` & `src/poly/engine.cpp`: NVFP4 global-scale binding (`*.scale` tensors).
- Reference math: `ggml-org/llama.cpp` PR `#32694` (`src/models/inkling2.cpp`) + PR `#32704`.

Implementation steps:
1. **`Inkling2::load_layer()`**:
   - Upload all non-expert tensors listed in Section 1.3 into per-physical-layer `Block` structs (`24` blocks).
   - Register fused `ffn_gate_up_exps.{weight,scale}` and `ffn_down_exps.{weight,scale}` with `Engine`'s expert store.
   - Note: `Engine` currently expects separate `ffn_gate_exps` and `ffn_up_exps` for some architectures; add a clean fused-`gate_up` path (or split view at expert-upload time) that preserves per-expert `.scale` on all 3 passes.
2. **`Inkling2::alloc_caches()` & `assign_scratch()`**:
   - Allocate **48 logical KV slots** (indexed by `step` in `0..47`): 8 global (`kv_alloc`, grows dynamically with `cover()`) + 40 SWA (`seq_alloc`, fixed `window_ + rows_` ring).
   - Allocate per-step shortconv state (`4 × 1024` for K/V + `4 × 2304` for attn/ffn post-conv) and block-6 Attention Residual history buffer.
3. **Unfused CUDA/HIP kernels in `src/poly/kernels.cu`** (simple, FP32-accumulated first):
   - Causal shortconv (`k=4`), QK/rel L2 clipping (`22.6274` / `8.0`), value norm without gamma.
   - Relative-position attention (`d_rel=16`, `rel_extent=1024`) + log-scaling (`n_floor=8192`, `alpha=0.1`) + learned sink (`attn_sink_proj`), for FP16 first, then Q8_0 KV.
   - Rank-64 loop-controller gates (`loop_{q,k,v,ffn}_ctrl_{down,forget,gain}` + `loop_msg_query`) and block-6 Attention Residuals (normalized values).
   - Router (`route_scale=8.0`, `exp_probs_b.bias`, top-8/128), situ softcap activation (`4.0` / `25.0`), and final logit softcap (`30.0`).
4. **Loop unrolling inside `Engine`**:
   - Either drive the 48-step schedule directly via `steps_` (mapping logical step `s -> phys = steps_[s].phys`) while sharing the 24 physical weight sets and 24 expert-cache pools.

### Phase 2 — Memory Placement & Offload Optimization (3–5× Win on 8 GB VRAM)
1. **Enable `--kv q8_0`**: Cuts 128K KV from 4.31 GiB → 2.29 GiB, returning ~2.0 GiB VRAM directly to the expert cache.
2. **Loop-core-weighted VRAM LRU expert cache**:
   - In `Engine`'s expert cache scoring, weight experts in core layers `6..17` by `loop_count_` (**3×**), since each cached core expert serves 3 passes per token instead of 1.
3. **Native NVFP4 execution on Blackwell (`sm_120`)**:
   - Route `UD-Q4_K_XL` / `UD-Q4_K_M` NVFP4 experts through `src/poly/mmq_nvfp4.cu` (FP4×FP4 tensor-core prefill) and `iq_mmvq` (`src/kernels/cuda/native_mmvq.cu`) for decode — zero FP16 dequantization buffers in VRAM.
4. **Profile with Nsight Systems (`nsys`)**: Capture 10 decode steps + 1 prefill batch to rank actual bottlenecks before fusing kernels.

### Phase 3 — Kernel Fusion & CUDA Graphs
Apply one fusion at a time, re-running Stage 2 greedy parity after **each** change:
1. **Attention prep fusion**: Fuse `shortconv (k=4)` + `QK/rel L2 clip` + `value norm` + `Q8_0 KV quantize-and-store` (cuts 6–8 launches/layer down to 1–2).
2. **Flash-decode fusion**: Fold `relative-position bias (d_rel=16)` + `log-scaling` + `learned sink` directly into the decode attention kernel (single VRAM pass over K/V).
3. **Residual & loop-gate fusion**: Fuse `rank-64 loop gate` + `attention residual (block 6)` + `post-conv` + `RMSNorm`.
4. **MoE activation fusion**: Since `ffn_gate_up_exps` is already fused in the GGUF, fuse `situ softcap (4.0/25.0)` + activation + input quantization for `ffn_down_exps` in registers.
5. **CUDA Graph capture**: Set `L.fixed = true` so `Engine::passes_` / `tail_` replays all 48 logical steps with minimal CPU launch overhead.

### Phase 4 — DSpark Speculative Decoding & Vision
1. **DSpark drafter** (`dspark-Inkling-2-Mini-Q8_0.gguf`, 0.4 GB): Implement the 3-layer Markov/RNN-hidden drafter (`mtp_config`: `block_size=8`, `markov_rank=256`, `target_layer_ids=[5, 23, 41, 48]`) via `Engine::draft_step()`.
2. **Vision**: Validate `mmproj-BF16.gguf` through `strata-vision` on image prompts.

### Phase 5 — Cross-Backend & Upstream Hygiene
- Build and verify **both CUDA and HIP** (`AGENTS.md`).
- Confirm all existing models remain byte-identical on default paths.
- Record measured RTX 5060 Laptop numbers in `docs/models/inkling2.md` and `bench/results/`.

---

## 4. 4-Stage Verification & Test Plan (Run on RTX 5060 Laptop)

### Stage 1 — Layer-by-Layer Tensor Parity (Bringup)
- Dump hidden states on a 1-token and 16-token prompt after:
  - `blk.0` (SWA + shortconv + situ MoE)
  - `blk.5` (1st global attention + `attn_k_conv_proj`)
  - `blk.6` **pass 1, pass 2, pass 3** (verifies loop gates, attention residuals, and per-pass NVFP4 scales!)
  - `blk.23` and final logits (`final_logit_softcapping = 30.0`)
- **Pass criteria**: Cosine similarity `>= 0.9999` vs `llama.cpp` (+ `#32704`) at every checkpoint; zero NaN/Inf.

### Stage 2 — Greedy Token-by-Token Parity
- Run with `temperature = 0`, `--no-draft` on both `UD-Q4_K_M` and `UD-Q4_K_XL`:
  - 5 short factual/coding prompts (must match `llama.cpp` token-for-token).
  - Prefixes of a 3,740-token prompt (100, 600, 1500, 3740 tokens — same method as Flint commit `5ff592f1`; any divergence must only occur where top-2 logits are within ~0.5 log-prob).

### Stage 3 — Q8_0 KV & Long-Context Verification
```powershell
# 1. FP16 vs Q8_0 KV agreement:
python tools/kv_agree.py ...

# 2. Needle recall at 32K, 64K, and 128K (verifies SWA ring wraparound >1024 and log-scaling >8192):
python tools/needle_bench.py --context 32768
python tools/needle_bench.py --context 65536
python tools/needle_bench.py --context 131072
```
- **Pass criteria**: Teacher-forced FP16/Q8_0 agreement `>= 97%`; 100% needle recall at 32K, 64K, and 128K.

### Stage 4 — Same-Day A/B Benchmark & Regression Suite
```powershell
# Same-day A/B speed & memory vs llama.cpp at 128K + Q8_0 KV:
python bench/bench_vs_llama.py ...

# Offline setup & server unit tests:
python tools/test_setup_inkling2.py
python tools/test_setup_groot.py
python tools/test_setup_menu.py
python tools/test_setup_config.py
python tools/test_setup_pins.py
python serve/test_models.py
python tools/test_strata_mcp.py
```
- **Pass criteria**: Faster than `llama.cpp` on 8 GB VRAM / 32 GB RAM offload; peak VRAM `<= 7.5 GiB`, peak RAM `<= 26 GiB`; all unit tests pass; existing `groot` (`qwen35moe`) greedy output unchanged.

---

## 5. GPU Portability Notes (One GPU vs All GPUs)

- **~80–90% of speedups transfer automatically to all GPUs (NVIDIA RTX 20–50 & AMD HIP)**:
  - Q8_0 KV cache, SWA ring buffers, 3× loop-core-weighted VRAM LRU expert cache, kernel fusion (fewer VRAM round-trips and launches across 48 logical layers), CUDA/HIP Graph capture, and DSpark speculative decoding.
- **~10–20% is generation/vendor specific**:
  - Native **FP4×FP4** tensor-core prefill (`src/poly/mmq_nvfp4.cu`) is Blackwell (`sm_120`, RTX 50) specific; RTX 20/30/40 fall back to their existing MMA/`dp4a` paths automatically.
  - AMD HIP wavefront (32 vs 64 lanes) and dot-product intrinsics use thin `#if defined(__HIPCC__)` guards inside shared kernel headers.
