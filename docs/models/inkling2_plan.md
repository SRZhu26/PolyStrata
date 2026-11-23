# Inkling 2 Mini 26B-A5B (`inkling2_loop`) — Implementation, Optimization & Test Plan

Self-contained engineering handoff for bringing up, optimizing, and verifying **Thinking Machines Lab's Inkling 2 Mini**
(`general.architecture = "inkling2_loop"`) in PolyStrata's `strata-poly` engine on a **Windows 11 laptop with an
NVIDIA RTX 5060 Laptop (8 GB VRAM, Blackwell `sm_120`) and 32 GB RAM**.

Branch: `add-inkling2-mini` is the pull-request head and must stay clean; the engine work and these adjustments
live on `inkling2-engine-local` (never push directly to `origin/main`).

> **Read [inkling2_progress.md](inkling2_progress.md) first.** Note: the earlier `llama.cpp` refusal in that report
> was from build `b11669`, which predates `inkling2_loop` (`ggml-org/llama.cpp` PR `#32694` + `#32704` commit
> `2a28df0f9b` or Unsloth `b12588-mix-c0c3de8`). For quick iteration on this RTX 5060 Laptop, use the focused
> `--kv q8_0` test plan in Section 0 below.

---

## 0. Quick Test Plan — RTX 5060 Laptop (8 GB VRAM, 32 GB RAM Offloading, `--kv q8_0`)

Focused verification & benchmark sequence after four fixes/optimizations landed on `fork/inkling2-engine-local`:
1. **`ink_conv` / `ink_conv_state` 256× redundant-thread & warp-race fix** (`src/poly/inkling2_kernels.cu`) — fixes
   intra-block warp data race on `state[]` when `rows < 3` and cuts 192 conv launches/call by ~100–256×.
2. **`Engine::read()` `n_bank_` leak/clobber fix** (`src/poly/engine.cpp:5538`: `!read_layer_major() ||`) — fixes
   `the depth attention's bank is full (5 of 5 at step 42)` when conversation marks split a read into `n_calls >= 2`.
3. **Parallel 32-lane warp softmax** in `ink_attend_rel_kernel` (replaces 128 serial `expf()` calls on `lane == 0`).
4. **Full `--kv q8_0` support** (`ink_attend_rel_q8` + `store_q8` + Q8_0 save/load in `src/poly/inkling2.cpp`) —
   cuts KV VRAM by 46.88% (128K KV drops from 4.31 GiB → **2.29 GiB**), but only frees it for experts when the KV is
   preallocated (`--kv-whole`): measured, 128K `--kv-whole` leaves **405** experts resident on Q8_0 and **0** on FP16. With
   lazy KV growth the pool claims first and Q8_0 buys 38 experts below 32K, at a 6–13% speed cost. See Step 4 results below.

### Step 1 — Rebuild & Kernel Unit Tests (~2 min)
```powershell
bash tools/build_poly_cuda_local.sh
.\build-poly\poly_inkling2_test.exe
```
Confirm clean CUDA `sm_120` build and all 17 FP64 kernel checks pass.

> **Windows build environment note (`reg.exe` blocked / empty `INCLUDE` & `LIB`)**:
> `vcvars64.bat` (`winsdk.bat`) invokes `reg.exe` to locate the Windows SDK. In an agent shell where `reg.exe` is
> blacklisted, `INCLUDE` and `LIB` stay empty and `cl.exe` fails (passing `-ccbin` alone does not populate them).
> - **Preferred (if `nvcc.exe` itself also invokes `reg.exe`)**: run the fast incremental rebuild directly in
>   **"x64 Native Tools Command Prompt for VS"** on the laptop:
>   ```powershell
>   git pull --ff-only fork inkling2-engine-local
>   cmake --build build-poly --config Release --target strata-poly poly_inkling2_test -j
>   .\build-poly\poly_inkling2_test.exe
>   ```
>   Once `build-poly\strata-poly.exe` is built, the agent can run all Python tests/benchmarks without `cl.exe`/`nvcc`.
> - **Alternative (skip `vcvars64.bat`, set `INCLUDE`/`LIB`/`PATH` from disk)**:
>   ```powershell
>   $msvc = (Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*" | Sort-Object Name -Descending | Select-Object -First 1).FullName
>   $sdk  = "C:\Program Files (x86)\Windows Kits\10"
>   $ver  = (Get-ChildItem "$sdk\Include" | Sort-Object Name -Descending | Select-Object -First 1).Name
>   $env:INCLUDE = "$msvc\include;$sdk\Include\$ver\ucrt;$sdk\Include\$ver\shared;$sdk\Include\$ver\um"
>   $env:LIB     = "$msvc\lib\x64;$sdk\Lib\$ver\ucrt\x64;$sdk\Lib\$ver\um\x64"
>   $env:PATH    = "$msvc\bin\Hostx64\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;$env:PATH"
>   cmake --build build-poly --config Release --target strata-poly poly_inkling2_test -j
>   ```

### Step 2 — Verify the Two Bug Fixes (~3 min)
1. **Bank-leak fix (`5 of 5 at step 42`)**: Run 6+ sequential requests on one server via `tools/probe_context.py`
   (with `--kv q8_0`) and confirm zero `bank is full` errors.
2. **Conv-state warp-race fix (intermittent corruption at ~10,900 tokens)**: Run 3 consecutive ~10,900-token
   prefills + 256-token decodes on a fresh server and verify all 3 produce coherent text (no repetition loop).
   **Done, with a correction:** the loop was never length-dependent. With one question held fixed and only the filler length
   changed, 6,185 / 8,233 / 10,281 tokens are all coherent; a README dump loops at 1,432 tokens under both KV formats and
   both greedy and sampled decoding. The loop follows the prompt's repeated proper noun, not the length. Coherence above 10.3K is
   still unmeasured. See `inkling2_progress.md` §8.1.

### Step 3 — End-to-End Parity & Q8_0 KV Agreement (~5 min)
```powershell
# 1. End-to-end FP64 cosine gate at blk.6, blk.23, and logits (cosine >= 0.999):
#    run with STRATA_DUMP_BIN + tools/dump_cos.py against tools/ref_core.py

# 2. FP16 vs Q8_0 KV teacher-forced agreement (>= 97%):
python tools/kv_agree.py `
  --model <path-to-UD-IQ4_XS-or-UD-Q4_K_XL.gguf> `
  --bin build-poly\strata-poly.exe `
  --ref fp16 --kv q8_0 --prompts 5 --max-new 128 --max-context 2048 --min-agree 0.97
```

### Step 4 — 8K / 32K / 128K Speed & Memory Sweep with `--kv q8_0`
Run on **`UD-IQ4_XS`** (15.73 GB, 0 non-NVFP4 scales, zero swap expected) and/or **`UD-Q4_K_XL`** (17.63 GB, for
direct before/after comparison against the earlier FP16 KV numbers of 10.1 tok/s decode & 71.7/39.4/23.4 tok/s prefill):
```powershell
# Generate 8K, 32K, and 128K prompts once (if not already in bench/):
python tools/make_long_prompt.py --out bench/prompt-8k.txt   --tokens 8000
python tools/make_long_prompt.py --out bench/prompt-32k.txt  --tokens 32000
python tools/make_long_prompt.py --out bench/prompt-128k.txt --tokens 126000

# 8K with Q8_0 KV (~0.30 GiB KV, max VRAM left for experts):
python tools/bench_128k_local.py --out bench/q8-8k   --models iq4xs=<gguf> --engines strata `
  --prompt bench/prompt-8k.txt   --context 8192   --kv q8_0 --gen 256

# 32K with Q8_0 KV (~0.70 GiB KV, ~3.7 GiB left for experts):
python tools/bench_128k_local.py --out bench/q8-32k  --models iq4xs=<gguf> --engines strata `
  --prompt bench/prompt-32k.txt  --context 32768  --kv q8_0 --gen 256

# 128K with Q8_0 KV (~2.29 GiB KV vs 4.31 GiB FP16, ~2.1 GiB left for experts):
python tools/bench_128k_local.py --out bench/q8-128k --models iq4xs=<gguf> --engines strata `
  --prompt bench/prompt-128k.txt --context 131072 --kv q8_0 --gen 256
```
Record for each length: **prefill tok/s, decode tok/s, TTFT, peak VRAM (<= 7,680 MiB), peak RSS, and swap (target 0 GiB)**.

### Step 4 — Results (2026-11-23), and what they change

Same binary, same prompt per pair, fresh server per row, `UD-Q4_K_XL`:

| Context | KV | Prefill tok/s | Decode tok/s | VRAM peak | RSS | Experts in VRAM |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8,198 | fp16 | **136.3** | **8.1** | 6,640 MiB | 12.46 GiB | 791 (3.8 GiB) |
| 8,198 | q8_0 | 127.7 | 7.2 | 6,640 MiB | 14.61 GiB | 829 (4.0 GiB) |
| 32,015 | fp16 | **77.3** | **8.7** | 6,640 MiB | 14.88 GiB | 791 |
| 32,015 | q8_0 | 66.9 | 7.6 | 6,640 MiB | 14.39 GiB | 829 |
| 128,072 (load only) | fp16 / q8_0 | — | — | — | — | 791 / 829 |
| 128,072, `--kv-whole` | fp16 | — | — | — | — | **0** |
| 128,072, `--kv-whole` | q8_0 | — | — | — | — | **405 (1.9 GiB)** |

**Three conclusions change this section's premise:**

1. **Q8_0 KV is correct** — teacher-forced agreement with FP16 is **99.79%** (479/480, 5 prompts × 96 tokens),
   above the 97% gate. It is a sound format.
2. **Q8_0 KV is slower, not faster** — 6.3% at 8K and 13.5% at 32K prefill, 11.1% and 12.6% decode. The saving
   below 32K is only 0.25–0.54 GiB, which buys 38 experts and does not pay for dequantizing K and V on every score and
   value element (`inkling2_kernels.cu:397`, `:432`).
3. **The 2.29 GiB / 4.31 GiB claim only holds with `--kv-whole`**, i.e. when the KV is preallocated for the whole
   context. There, 128K FP16 leaves **0** experts resident and Q8_0 leaves **405**. With lazy KV growth the expert pool claims
   the card first and Q8_0 buys only the 38 experts its 0.25–0.54 GiB covers.

So Step 4's expected win does not exist at 8K/32K. Revised Step 4: **keep FP16 KV as the default and measure Q8_0 only
with `--kv-whole` at ≥128K**, where residency is the binding constraint. Also pin `--ram-experts-mib` whenever Q8_0 is
used: the auto heuristic flips to the unpinned path on the freed VRAM (see `inkling2_progress.md` §8.2), costing 0.4 tok/s
and 2.5 GiB of RSS at 8K.

A faster Q8_0 path is a kernel problem, not a format problem: an `__dp4a` integer dot over the per-32-block scales, as
`quantize.cpp`'s vec dot does, instead of `__half2float` per element.

### Step 4 — Re-measured after `f159fe78` (2026-11-23 09:44, FP16 KV)

The `f159fe78` attention rewrite (Q cached in shared memory, one dimension per thread in the value pass, hoisted Q8_0
scales, one Q8_1 quantization shared by `wq`/`rel`/`w_gate`/`sink_proj`) was built with
`tools/build_poly_cuda_local.sh` (`BUILD_OK`) and all 17 FP64 kernel checks pass.

| Context | Prefill (before → after) | Decode (before → after) | RSS (before → after) |
| ---: | --- | --- | --- |
| 8,198 | 136.3 → **171.3** (+25.7%) | 8.1 → **10.3** (+27.2%) | 12.46 → 11.20 GiB |
| 32,015 | 77.3 → **97.5** (+26.1%) | 8.7 → **9.5** (+9.2%) | 14.88 → 11.74 GiB |

Same 791 experts resident and the same 6,640 MiB VRAM in every row. The 8K pair is a clean A/B (both pinned a RAM pool);
the 32K pair is confounded, because the `ee04f3d7` run took the unpinned path (defect §8.2) and the new one pinned
10,319 MiB. This moves the decode baseline from 8.1 to **10.3 tok/s at 8K** and from 8.7 to **9.5 at 32K**; the
`1.5×` vs `llama.cpp` gate is still undefined (no baseline exists), so §4's absolute floors should be re-based on these.

### Step 5 — Next High-Impact Speedups (After Step 4 Numbers)
1. **Enable `read_layer_major()`** (move per-call depth-bank/message scratch to host): eliminates re-reading experts
   on every 1,024-token chunk (**4–10× faster 32K/128K prefill**).
2. **Attention kernel upgrade**: MMA tensor-core prefill + split-K flash-decoding for `ink_attend_rel_q8`, and `__dp4a` for
   the integer dot, without which Q8_0 KV cannot beat FP16 at any length measured.
3. **Pass-aware `tail_key()` + CUDA Graphs (`L.fixed = true`)** + kernel fusion for decode.

---

## 1. Verified Model & GGUF Ground Truth

Extracted directly from the live GGUF headers of `unsloth/Inkling-2-Mini-GGUF` at pinned revision
`ea0c79b2ecd73669e774c8519b79bfabfddbc027` (`Mainline_Support` layout, matching merged `ggml-org/llama.cpp` PR `#32694`):

### 1.1 Pinned Files, Exact Sizes & Quality vs `UD-Q8_K_XL`

| Quant | File Size | Mean KLD | Same Top Token | Routed Experts (`arena_gb`) | Non-Expert Weights | Gate+Up / Down Expert Types | Non-NVFP4 `.scale` Layers |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| **UD-IQ4_XS** *(provisional default)* | 15.73 GB | 0.0109 | 97.7% | 14.04 GB (13.07 GiB) | 1.69 GB (1.57 GiB) | `IQ4_XS, NVFP4, Q4_K, Q5_K` / `IQ4_NL, NVFP4, Q5_0` | **0** |
| **UD-Q3_K_XL** *(speed/low-RAM)* | 13.63 GB | 0.0174 | 96.3% | 12.20 GB (11.36 GiB) | 1.42 GB (1.32 GiB) | `IQ2_S, IQ2_XS, IQ3_S, IQ3_XXS, NVFP4, Q3_K` / `IQ4_NL, NVFP4` | **8** |
| UD-IQ3_XXS *(smallest Q3)* | 11.66 GB | 0.0269 | 94.7% | 10.30 GB (9.59 GiB) | 1.35 GB (1.26 GiB) | `IQ2_S, IQ2_XS, IQ2_XXS, IQ3_XXS` / `IQ4_NL, NVFP4` | **20** |
| UD-Q4_K_XL *(bringup ref)* | 17.63 GB | 0.0107 | 98.0% | 15.80 GB (14.71 GiB) | 1.83 GB (1.71 GiB) | `NVFP4, Q8_0` / `NVFP4, Q8_0` | **0** |
| UD-Q4_K_M | 16.61 GB | 0.0108 | 97.9% | 14.78 GB (13.76 GiB) | 1.83 GB (1.71 GiB) | `NVFP4, Q5_K, Q8_0` / `NVFP4, Q8_0` | **0** |
| `mmproj-BF16.gguf` | 0.64 GB | — | — | — | 0.64 GB | `BF16` | — |

- **Non-expert weight types depend on the quant**: `Q8_0/BF16/F32` on the two `Q4_K` files, adding `Q5_K/Q6_K` on
  `UD-IQ4_XS` and `Q4_K/Q5_K/Q6_K` on `UD-Q3_K_XL` / `UD-IQ3_XXS`. Load every non-expert matrix through
  `upload(..., kAnyMat)`.
- **Down projections (rows of 1,152 = 4.5 × 256)** cannot use 256-value block types, so they use 32- or 64-value
  block types (`IQ4_NL`, `Q5_0`, `NVFP4`, `Q8_0`). `native_mmvq.cu` and `mmq.cu` already support all of these types.
- **Q3 non-NVFP4 `.scale` finding (verified from headers)**: `UD-IQ4_XS`, `UD-Q4_K_M`, and `UD-Q4_K_XL` have **0**
  non-NVFP4 `.scale` tensors. `UD-Q3_K_XL` has **8** and `UD-IQ3_XXS` has **20** (`blk.*.ffn_gate_up_exps.scale`
  on `IQ2_S`/`IQ2_XS`/`IQ2_XXS`). Because both `Engine::expert_scales()` (`engine.cpp:3571`) and the CPU path
  (`engine.cpp:4106-4114`) are already type-agnostic, `Inkling2::load_layer()` can bind those scales for any type
  after `load_moe(il, false)` with zero shared-engine changes if their values are not 1.0.
- **Bringup order**:
  1. **`UD-Q4_K_XL`** (simplest: pure `NVFP4`/`Q8_0` experts) to prove Stage 1 & Stage 2 math.
  2. **`UD-IQ4_XS`** (adds `IQ4_XS`/`Q4_K`/`Q5_K` gate+up, `Q5_0`/`IQ4_NL` downs, `Q6_K` attention/head; 0 non-NVFP4 scales).
  3. **`UD-Q3_K_XL`** (exercises non-NVFP4 `.scale` binding and int8 `mmq` prefill; measure prefill separately since FP4×FP4 only helps NVFP4 layers).
  4. **`UD-IQ3_XXS`** (optional).
- Lock in the final default (`UD-IQ4_XS` vs `UD-Q3_K_XL`) only after Stage 4 benchmarks on the laptop (`AGENTS.md`).

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

### Phase 0 — Reference Oracle & Header Audit on RTX 5060 Laptop
1. Install Visual Studio Build Tools (C++), CUDA Toolkit (12.8 or 13.0), and CMake.
2. Checkout this branch and download the target files (`UD-Q4_K_XL` for Stage 1 bringup, then `UD-IQ4_XS` and
   `UD-Q3_K_XL` for benchmarking):
   ```powershell
   git fetch fork
   git checkout add-inkling2-mini
   python setup.py --family inkling2 --model UD-Q4_K_XL --vision no --yes --no-start
   ```
   Run a per-layer header audit with `tools/gguf_reader.py` on each downloaded file (recording `gate_up`/`down`
   types, `.scale` min/max values on any non-NVFP4 layers, and attention/head types).
3. **Confirm the `llama.cpp` question once, and record the answer.** Build `b11669` (commit `620d2225c`) and point it at
   the file:
   ```powershell
   llama-cli.exe -m ..\models\inkling2-mainline\Inkling-2-Mini-UD-Q4_K_XL.gguf -n 1 -p "hi"
   ```
   It answers `unknown model architecture: 'inkling2_loop'`, and neither `llama-server.exe` nor `llama.dll` contains the
   string `inkling`. **Mainline `llama.cpp` has no `inkling2` implementation**, so there is no speed baseline and no greedy
   reference to obtain — porting one would mean writing the same ~2,000 lines, including the nine kernels listed in
   [inkling2_progress.md](inkling2_progress.md) §3. Do not re-litigate this per session; re-run the command only if a
   newer `llama.cpp` claims support.
4. **Use the FP64 oracle as the reference instead.** `tools/ref_core.py` runs the whole model — all 74 stages plus logits —
   in double precision from the same GGUF, and `tools/ref_prelude.py` is the validated per-stage version. This replaces the
   `llama-bench` speed baseline and the `llama-cli --temp 0` greedy reference the earlier draft of this plan called for.
   - Prefill and decode are then measured absolutely (§4 below), not as a ratio.
   - The `≥ 1.5×` decode gate is replaced by the VRAM, RAM and absolute-throughput gates in Stage 4.

### Phase 1 — Exact Unfused Forward Pass (`src/poly/inkling2.cpp`)

**Status: implemented, not yet verified.** The kernels and overrides exist and produce coherent text; the exit
criterion (Stage 1 cosine) has not been run. See [inkling2_progress.md](inkling2_progress.md) §3 and §7.

- **Math reference**: the formula list below is the specification (relative-position attention, QK/rel L2 clip,
  global-only log-scaling, value norm without gamma, shortconv `k=4`, global K/V conv taps, rank-64 loop gates, block-6
  attention residuals, router + `exp_probs_b.bias`, situ softcaps `4.0`/`25.0`, 966-row vocab padding mask,
  `final_logit_softcapping = 30.0`). PR `#32704`'s `src/models/inkling2.cpp` remains a cross-check for the formulas, but
  its build cannot load the file (Phase 0 step 3), so it cannot be used as an executable reference.
- **PolyStrata code templates**:
  - `src/poly/flint.cpp`: hybrid SWA ring buffers (`ring_ = std::min(opt_.max_context, window_ + rows_)` inside
    `alloc_caches()`, **never** `load_geometry()`), Q8_0 KV, `save_layer()` / `load_extra()`.
  - `src/poly/lfm2moe.cpp`: causal shortconv (`l_cache = 4`) recurrent state.
- **Keep every kernel simple and FP32-accumulated** in Phase 1; do not enable Blackwell FP4×FP4 activation
  quantization or kernel fusion until Stage 1 & Stage 2 pass.

### Phase 1a — Defects found during bringup (close before Phase 2)

These were not in the original plan; they are now the gating work. Detail in
[inkling2_progress.md](inkling2_progress.md) §8.

1. **Intermittent long-context corruption** — ~2 of 3 prefills near 10,900 tokens degenerate into a repetition loop,
   while the same binary on the same prompt sometimes stays coherent. A race, proven by a coherent 10,890 vs a degenerate
   10,898. Suspects left: `moe_rows()`'s `kStageRows = 32` staging and `copy_seq()`'s per-chunk recurrent state.
2. **Depth-attention bank leak** — the 6th sequential request dies with `bank is full (5 of 5 at step 42)`; one state leaks
   per request against `max_states_ = 5`.
3. **CUDA-graph invalidation gap** — `shrink_cache()` / `resize_card_cache()` move experts between slots without
   invalidating graphs except under `STRATA_GPU_METAL`. Latent while `L.card == 1` is false.
4. **Run the end-to-end cosine gate** — `STRATA_DUMP_BIN` + `tools/dump_cos.py` vs `tools/ref_core.py` at `blk.6`,
   `blk.23` and the logits. Kernel parity (`poly_inkling2_test`, 17 checks) is not a substitute.
5. **Redefine the speed gate** — the `llama.cpp` ratio is undefined; use the absolute floors in Stage 4.

### Phase 1b — Tokenizer, Chat Template & Server (Required Before Stages 3 & 4)
1. Read `tokenizer.ggml.pre` from the GGUF. If its pre-tokenizer pattern is not already in `strata_tokenizer.py`,
   add it to `PRE_PATTERNS`.
2. Encode 20 mixed strings (code, CJK, emoji, numbers) and compare token IDs byte-for-byte against the Hugging Face
   tokenizer for this model. *(Not against `llama-tokenize`: that build cannot load the file — Phase 0 step 3.)*
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

**Status: the cosine half of this stage is unrun; the kernel half already passes.** `poly_inkling2_test` compares each
kernel against an FP64 reference of the same graph and its 17 checks pass, but that is not the assembled-model check below.

- **Reference**: `tools/ref_core.py`, this repository's own FP64 reference, which runs all 74 stages plus logits over the
  exact decoded GGUF blocks. `tools/ref_prelude.py` is the validated per-stage version and `tools/ref_blk0.py` /
  `tools/ref_expert.py` are narrower ones. *(The earlier draft named ninfer-all's `tools/reference/inkling2.py`; that file is
  not in this repository.)*
- **Per-visit check** (add a temporary `STRATA_INKLING2_DUMP=1` `cudaMemcpy` hook at the end of `Inkling2::layer_tail()`):
  - Cosine similarity **`>= 0.999`** vs FP64 after `blk.0`, `blk.5`, `blk.6` on **passes 1, 2, and 3**, `blk.23`, and
    the final logits.
  - Zero NaN/Inf; passes 2 and 3 no worse than pass 1 (confirms per-expert NVFP4 scales stay bound).
- **NLL / perplexity check**:
  - Mean NLL within **0.1%** of FP64 on ~1,150 WikiText tokens (past the 1,024 SWA window).
  - One run past **8,192 tokens** to exercise global-layer log-scaling.
  - *(Calibration reference: ninfer-all measured `+0.03%` vs FP64. There is no `llama.cpp` perplexity to compare
    against, so the earlier `-0.06%` target is dropped.)*

### Stage 2 — Per-Quant Greedy Token Parity & NLL vs FP64

There is no `llama.cpp` to compare against on the same file (see Phase 0 step 3), so parity is measured against the
oracle and across quants instead:

- Run with `temperature = 0`, `--no-draft` on each target quant (`UD-Q4_K_XL`, `UD-IQ4_XS`, `UD-Q3_K_XL`, optional
  `UD-IQ3_XXS`) and compare against `tools/ref_core.py` **on the same file**:
  - Mean NLL within **0.1%** of the FP64 reference on that file.
  - Greedy top-1 agreement with the FP64 reference of at least **~97%** over 5 × 256 steps (plus exact or near-tie match
    on 5 short prompts and prefixes of a 3,740-token prompt, printing top-3 logits at any first divergence).
  - Cross-quant sanity: `UD-Q4_K_XL` vs `UD-IQ4_XS` top-1 agreement within ~1 point of Unsloth's published
    `98.0 / 97.7 / 96.3 / 94.7%` figures, which is the only external calibration available.

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
`bench/inkling2_machine.json` carries a **single engine**, because there is no `llama.cpp` side to benchmark against on
this architecture. The A/B is across **KV types and quants**, not across engines:
```json
{
  "gguf": "models\\inkling2-ud-q4_k_xl\\Inkling-2-Mini-UD-Q4_K_XL.gguf",
  "strata_cfg": "strata-inkling2-ud-q4_k_xl.json",
  "configs": [
    {"name": "fp16-kv", "kind": "strata", "args": []},
    {"name": "q8_0-kv", "kind": "strata", "args": ["--kv", "q8_0"]}
  ]
}
```
```powershell
# 1. Same-day A/B benchmark (bench_vs_llama.py has 4K and 32K cells; measure 64K/128K via needle_bench or added cells):
python bench/bench_vs_llama.py prompts --out bench/inkling2_vs_llama --repo . `
  --tokenizer <pack>\tokenizer --corpus <text file of 40K+ tokens>
python bench/bench_vs_llama.py run --machine bench/inkling2_machine.json --out bench/inkling2_vs_llama `
  --configs fp16-kv,q8_0-kv --rounds 3
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
- **Pass criteria** (revised: no `llama.cpp` ratio exists, so these are absolute and relative to the measured baseline
  in [inkling2_progress.md](inkling2_progress.md) §4):
  - Benchmark **`UD-IQ4_XS`** and **`UD-Q3_K_XL`** at **32K with Q8_0 KV** (including peak RAM), and choose the final
    `setup.py` default from those measured numbers.
  - **Peak VRAM `<= 7.5 GiB`, peak RAM `<= 26 GiB`** — both currently pass, narrowly (7,498 MiB and 25.2 GiB at 8K).
  - **No regression against the current baseline**, measured at FP16 KV on `UD-Q4_K_XL`: prefill `>= 71.7 tok/s` at 8K
    and `>= 84.4 tok/s` at 16K; decode `>= 10.1 tok/s` at 16K. Q8_0 KV must be no worse than FP16 within noise,
    which is the actual point of this stage.
  - Once `read_layer_major()` is enabled (§9 of the progress report), these floors are expected to move substantially and
    should be re-baselined rather than treated as a ceiling.
  - All unit tests pass; existing `groot` (`qwen35moe`) greedy output unchanged.

---

## 6. GPU Portability Notes (One GPU vs All GPUs)

> **Correction after implementation.** The transfer claimed below applies to the *shared* engine code, not to the
> `inkling2` kernels. `src/poly/inkling2_kernels.cu:28` hardcodes `cudaStream_t` with no `__HIP__` branch (sibling
> `kernels.cu` has one), and there is no Metal variant, so **an AMD or Apple build of this model does not compile today.**
> Porting the nine kernels in [inkling2_progress.md](inkling2_progress.md) §3 is the prerequisite, and it is deliberately out
> of scope for now. CPU-only is also unsupported: `strata-poly` is only built under `CUDA OR HIP`.

- **~80–90% of the optimization work is backend-independent** (it transfers once the kernels are ported):
  - Q8_0 KV cache, SWA ring buffers, VRAM LRU expert cache (naturally counting core experts 3× per token), kernel
    fusion, and CUDA/HIP Graph capture.
- **~10–20% is generation/vendor specific**:
  - Native **FP4×FP4** tensor-core prefill (`src/poly/mmq_nvfp4.cu`) is Blackwell (`sm_120`, RTX 50) specific;
    RTX 20/30/40 fall back to their existing MMA/`dp4a` paths automatically.
  - AMD HIP wavefront (32 vs 64 lanes) and dot-product intrinsics need thin `#if defined(__HIPCC__)` guards in the
    `inkling2` kernels, which they do not have yet.

---

## 7. 128K Agentic Coding Optimization Playbook (RTX 5060 Laptop — 8 GB VRAM, 32 GB RAM)

Agentic coding (Cursor / Claude Code / Cline / pi / OpenHands) has a different profile from synthetic 128K single-turn
benchmarks: Turn 1 prefills the system prompt + repo context (4K–16K), and Turns 2..50 are incremental appends of
200–2,000 tokens reusing the cached KV prefix (`--pinned-session` / conversation cache). However, at **128K filled
context** on an 8 GB VRAM card, **FP16 KV takes 4.31 GiB and evicts 100% of experts from VRAM (`0 of 3,072`)**,
whereas **Q8_0 KV takes 2.29 GiB and keeps ~405–550 experts (~1.9–2.1 GiB) resident**.

The goal is therefore: **make `--kv q8_0` as fast as or faster than FP16**, maximize VRAM expert hit rate, and remove
the remaining prefill/decode walls.

### 7.1 `__dp4a` Integer Dot Product in `ink_attend_rel_q8_kernel` *(Implemented)*
- **What changed** (`src/poly/inkling2_kernels.cu`, includes `include/strata/kernels/dp4a.hpp`):
  - Before the key-tile loop, each warp quantizes its block's `nr` queries into shared memory (`__shared__ int qq4[8][64]`
    and `__shared__ float qscale[8][8]`, 32-element blocks, using **8× less shared memory** than FP32 `qs`).
  - Inside the `Q·K` loop, each lane loads one 32-bit packed word (`4 × int8_t`, 128-byte coalesced warp transaction
    for `dim = 128`), executes **`STRATA_DP4A(qr4[w], kr4[w], 0)`** (1 instruction for 4 MACs), multiplies by
    `qscale[r][b] * __half2float(ksr[b])`, and reduces across the warp.
  - For `dim = 128`, the `Q·K` inner loop runs **1 iteration per key** instead of 4 scalar iterations.
- **Verify after rebuild**:
  ```powershell
  # 1. Rebuild (in x64 Native Tools Command Prompt for VS if reg.exe is blocked in agent shell):
  cmake --build build-poly --config Release --target strata-poly poly_inkling2_test -j
  .\build-poly\poly_inkling2_test.exe

  # 2. Verify Q8_0 vs FP16 agreement (>= 97%) after int8 query quantization:
  python tools/kv_agree.py --model <gguf> --bin build-poly\strata-poly.exe `
    --ref fp16 --kv q8_0 --prompts 5 --max-new 128 --max-context 2048 --min-agree 0.97

  # 3. Clean A/B with pinned RAM pool (--ram-experts-mib avoids the §8.2 auto -1 flip):
  python tools/bench_128k_local.py --out bench/dp4a-q8-8k  --models q4xl=<gguf> --engines strata `
    --prompt bench/prompt-8k.txt  --context 8192  --kv q8_0 --ram-experts-mib 10240 --gen 256
  python tools/bench_128k_local.py --out bench/dp4a-q8-32k --models q4xl=<gguf> --engines strata `
    --prompt bench/prompt-32k.txt --context 32768 --kv q8_0 --ram-experts-mib 10240 --gen 256
  ```

### 7.2 Expert Caching, Swapping & Quantization Levers (Highest Decode ROI — Zero C++ Changes)
At ~10 tok/s decode on `UD-Q4_K_XL`, ~65–70% of decode time is cold expert streaming from RAM/swap (only 791 of 3,072
experts = 25.7% resident).

1. **Switch from `UD-Q4_K_XL` (17.63 GB) → `UD-IQ4_XS` (15.73 GB, KLD 0.0109, 0 non-NVFP4 scales)**:
   - Eliminates the 0.7 GiB Windows swap, reduces RAM pressure by ~2 GB, fits ~12–15% more experts in VRAM, and reads
     11% fewer bytes per CPU miss.
   - For maximum speed, also test `UD-Q3_K_XL` (13.63 GB; note 8 non-NVFP4 `.scale` layers — bind them in
     `Inkling2::load_layer()` after `load_moe(il, false)` if their values are != 1.0).
2. **Warm-seed the VRAM cache with a coding routing profile (`--profile`)**:
   - Inkling's core layers `blk.6..17` run 3× per token so `Engine::use_[]` naturally weights them 3×.
   - Record a coding session profile (`--profile bench/inkling2_code.profile`) and pass it on server startup so the
     8 GB VRAM cache starts pre-populated with hot coding experts instead of warming up from scratch (ninfer-all
     reached **76.7%** VRAM hit rate and **33.1 tok/s** on desktop RTX 5060 8 GB with `UD-IQ4_XS` + profile).
3. **Always pin `--ram-experts-mib` (e.g. `10240`) and test `--vram-pool` + `--pinned-session`**:
   - `--ram-experts-mib 10240` avoids the §8.2 knife-edge flip to `-1` (keeps the pinned RAM pool + 8 prefetch threads).
   - `--vram-pool` lets KV and the expert cache trade pages cleanly inside one pool as a 128K session grows.
   - `--pinned-session` keeps the coding agent's system prompt + tool definitions pinned across turns.
4. **Optional `--skip-weight` tuning**:
   - With 384 expert evaluations/token (48 visits × top-8), test small `--skip-weight` values (e.g. `0.02`–`0.05`) on
     coding prompts to skip tiny-weight cold CPU misses while checking that coding accuracy does not regress.

### 7.3 Split-K Flash-Decoding for `rows == 1` at 64K–128K (How to Implement)
- **Bottleneck**: During decode (`rows = 1`, `nr = 1`), `ink_attend_rel` / `ink_attend_rel_q8` launches only
  **24 blocks total** (`gridDim = (24, 1)`), leaving most RTX 5060 SMs idle while each head scans up to 131,072 keys
  serially.
- **How to implement** (`src/poly/inkling2_kernels.cu` + `src/poly/inkling2.cpp`):
  1. In `Inkling2::assign_scratch()`, allocate a tiny split scratch buffer:
     - `constexpr int kAttnSplits = 16;`
     - `split_out_ = fbuf((size_t) kAttnSplits * n_head_ * head_dim_);` (192 KB)
     - `split_ml_  = fbuf((size_t) kAttnSplits * n_head_ * 2);` (`max`, `sum` per split, 3 KB)
  2. When `rows == 1 && !B.swa && pos >= 2048`:
     - **Kernel 1 (`ink_attend_rel_q8_split_kernel`)**: Launch `dim3(n_head, kAttnSplits)` (384 blocks!). Block `(h, sp)`
       scans keys `[sp * chunk, min((sp + 1) * chunk, pos + 1))` using the exact same `__dp4a` inner loop (with the
       sink's `m = 0, sum = 1` applied **only** on `sp == 0`, and `m = -INFINITY, sum = 0` on `sp > 0`), writing
       `(m_sp, sum_sp)` to `split_ml_` and unnormalized `acc_sp` to `split_out_`.
     - **Kernel 2 (`ink_attend_split_reduce_kernel`)**: Launch `n_head` blocks of 128 threads; compute global
       `m_all = max_sp(m_sp)`, rescale each split by `expf(m_sp - m_all)`, sum `acc_sp` and `sum_sp`, write `out`.
  3. **Verify**: Compare split output vs non-split output in `poly_inkling2_test` at `pos = 4096` (max abs diff `< 1e-5`).

### 7.4 Enabling `read_layer_major()` (3–6× Faster 32K/128K Prefill — How to Implement)
- **Bottleneck**: With `read_layer_major() == false`, a 128K prompt runs 128 separate 1,024-token `eval()` calls,
  re-reading all 14–16 GB of experts from RAM **128 times** (~2 TB of traffic).
- **How to implement**:
  1. In `Inkling2`, each call's cross-layer state (beside `r_`) is `bank_` (`5 × rows × 2304 × 4 B ≈ 45 MB`),
     `prev_bank_` (`45 MB`), and `prev_out_` (`9 MB`) — ~100 MB per call at `rows = 1024`.
  2. In `Engine::read()` (or an `Inkling2` hook called around `layer(il, call.m)` in `read()`), allocate a pinned host
     slot per call (`n_calls` up to e.g. 4–8 calls = 400–800 MB host RAM) to save/restore `(bank_, prev_bank_, prev_out_,
     n_bank_, n_prev_bank_)` alongside `r_` on the async copy stream `rs_`.
  3. Return `true` from `Inkling2::read_layer_major()` (capping `read_rows_` to e.g. `4 * rows_` = 4,096 tokens so host
     staging stays under 400 MB) and verify identical output vs `read_layer_major() == false` on an 8,192-token prompt.

### 7.5 Tensor-Core MMA Prefill + Pass-Aware CUDA Graphs (`L.fixed = true`)
1. **Pass-aware CUDA Graphs (`L.fixed = true`)**:
   - Add `virtual int tail_key() const { return 0; }` on `Engine` (`steps_[cur_step_].pass` in `Inkling2`), key
     `L.tail` / `L.shared` / `L.card_moe` by `rec + tail_key() * recs`, and combine `head_key` with `room_key(n)` on
     global layers as `pass * (max_room_key + 1) + room_key(n)`.
   - **Close defect §8.3 first**: invalidate recorded CUDA graphs (`passes_.clear()` / destroy `cudaGraphExec_t` handles
     in `L.heads`, `L.tail`, `L.card_moe`) inside `shrink_cache()` and `resize_card_cache()` on CUDA (currently only
     done under `#if defined(STRATA_GPU_METAL)`).
2. **Tensor-Core MMA Prefill Attention**:
   - Replace scalar FP32 `Q·K` / `P·V` in `ink_attend_rel` for `rows >= 64` with a FlashAttention-style MMA tile
     (following `src/poly/fattn_mma.cuh`), folding the `kqb` relative-position bias into the `S` accumulator tile
     before softmax.

### 7.6 Agentic Correctness Gates (Before Daily-Driving on Real Repos)
1. **End-to-end cosine gate on `UD-Q4_K_XL`**: `UD-Q4_K_XL` has **zero `Q4_K` tensors** (pure `NVFP4`/`Q8_0` experts
   and `Q8_0`/`BF16`/`F32` non-experts), so `STRATA_DUMP_BIN` + `tools/dump_cos.py` vs `tools/ref_core.py` is **not**
   blocked — run it now to confirm cosine `>= 0.999` at `blk.6`, `blk.23`, and final logits.
2. **Tokenizer parity**: Verify `tools/strata_tokenizer.py` (`GPT4O_PATTERN`) byte-for-byte on 20 mixed coding strings
   (Python/C++/TS, JSON, Unicode, tabs/indent) against `transformers` or `llama-tokenize`.
3. **Tool-call & reasoning parsing**: Add tests in `serve/test_models.py` confirming `<|content_thinking|>` maps to
   `reasoning_content` and `<|content_invoke_tool_json|>` maps cleanly to OpenAI `tool_calls`.

---

## 8. Detailed Next-Step Execution Plan (Step-by-Step for the RTX 5060 Laptop)

Current baseline on `inkling2-engine-local` (`UD-Q4_K_XL`, `--kv q8_0 --ram-experts-mib 10240`, Split-K passing):
- **8K**: 165.0 tok/s prefill, 10.1 tok/s decode, 6,640 MiB VRAM
- **32K**: 94.9 tok/s prefill, 9.1 tok/s decode, 6,640 MiB VRAM
- **130.7K**: 35.5 tok/s prefill (61.3 min TTFT), 6.7 tok/s decode, 6,640 MiB VRAM, 0.54 GiB swap

### Step 1 — Commit & Push Working Split-K + 128K Results (Keep `read_layer_major() = false`)
Do **not** hold back working Split-K behind the broken `read_layer_major()` path.
1. Ensure `Inkling2::read_layer_major()` returns `false` (and keep any unfinished layer-major edits stashed or on a
   separate WIP branch so `inkling2-engine-local` stays 100% green).
2. Run `.\build-poly\poly_inkling2_test.exe` to confirm all checks (including `attend split` and `attend q8 spl`) pass.
3. Commit and push **only** to `fork/inkling2-engine-local` (never `origin/main` or `add-inkling2-mini`):
   ```powershell
   git add src/poly/inkling2* include/strata/poly/kernels.hpp docs/models/inkling2_progress.md docs/models/inkling2_plan.md
   git commit -m "poly/inkling2: add Split-K flash decoding and 8K/32K/128K Q8_0 benchmark results"
   git push fork inkling2-engine-local
   ```

**Done** — `122cc988`, merged with this plan's `b0e88f70`+`412dc796`. The 19 checks pass (`attend split` 3.278e-07 of
0.449, `attend q8 spl` 2.086e-07 of 0.423); the layer-major attempt is on `inkling2-layer-major-wip` (`bb51e87a`) and
`read_layer_major()` returns `false`. The pass-restart fix (every pass after the first starts from the prelude's output) went in
with it, as did the end-to-end cosine gate (§4.5, `blk.6` 0.999988758).

### Step 2 — Maximize Dedicated dGPU VRAM Utilization + `UD-IQ4_XS` + `--expert-profile` (Biggest Immediate Decode Win)
**Why VRAM stopped at 6,640 / 8,151 MiB (~1,511 MiB unused):**
On this laptop, Windows desktop/display runs on the Intel Core Ultra 9 275HX iGPU, so the RTX 5060 dGPU (8,151 MiB)
is dedicated to compute. However, `strata-poly` reserves 512–1,024 MiB by default (`default_reserve_mib()` in
`src/poly/engine.cpp:1888`) when `ram_tight_` is true, AND DXGI caps the process budget conservatively.

**How to reclaim ~768–1,100 MiB of idle GDDR7 VRAM (+160–250 more resident experts):**
- Pass **`--vram-reserve-mib 256`** (the engine's safe floor — leaves 256 MiB for CUDA driver, cuBLAS workspace, and
  future CUDA graphs; do not go below 256 MiB or CUDA allocations can OOM).
- Pass **`--vram-pool`** so KV and the expert cache share one pre-claimed VRAM pool cleanly without VMM thrashing.
- Pass **`--ram-experts-mib 10240`** (avoids the §8.2 auto `-1` unpinned flip on Q8_0).
- Switch from `UD-Q4_K_XL` (17.63 GB, 0.54 GiB swap) to **`UD-IQ4_XS`** (15.73 GB, KLD 0.0109, 0 non-NVFP4 scales,
  expected 0 GiB swap, ~12–15% more experts fit in VRAM).
- Warm-seed the VRAM cache with a coding routing profile (**`--expert-profile bench/inkling2-code.profile`** — note
  the exact flag name in `src/poly/main.cpp:1562` is `--expert-profile`, not `--profile`).

**Exact commands to run on the RTX 5060 Laptop:**
```powershell
# 1. Download UD-IQ4_XS (if not already on disk):
python setup.py --family inkling2 --model UD-IQ4_XS --vision no --yes --no-start

# 2. Warm-up & record a coding routing profile (run 2-3 real coding prompts on a server started with
#    --expert-profile bench/inkling2-code.profile --vram-reserve-mib 256 --vram-pool --ram-experts-mib 10240):
python tools/bench_128k_local.py --out bench/iq4xs-8k-prof --models iq4xs=<path-to-UD-IQ4_XS.gguf> --engines strata `
  --prompt bench/prompt-8k.txt --context 8192 --kv q8_0 --ram-experts-mib 10240 --gen 256 `
  --strata-args --vram-reserve-mib 256 --vram-pool --expert-profile bench/inkling2-code.profile

# 3. Measure 8K, 32K, and 128K with the warm profile + reclaimed VRAM:
python tools/bench_128k_local.py --out bench/iq4xs-32k-prof --models iq4xs=<path-to-UD-IQ4_XS.gguf> --engines strata `
  --prompt bench/prompt-32k.txt --context 32768 --kv q8_0 --ram-experts-mib 10240 --gen 256 `
  --strata-args --vram-reserve-mib 256 --vram-pool --expert-profile bench/inkling2-code.profile

python tools/bench_128k_local.py --out bench/iq4xs-128k-prof --models iq4xs=<path-to-UD-IQ4_XS.gguf> --engines strata `
  --prompt bench/prompt-128k.txt --context 131072 --kv q8_0 --ram-experts-mib 10240 --gen 256 `
  --strata-args --vram-reserve-mib 256 --vram-pool --expert-profile bench/inkling2-code.profile
```
**Check engine logs** (`bench/.../engine-*.log`) for:
- Resident experts count (should jump from `791` → **`~1,050–1,200+`** at 8K/32K and `405` → **`~600+`** at 128K!).
- Peak VRAM (should climb from `6,640 MiB` → **`~7,400–7,800 MiB`** cleanly without OOM).
- Swap peak (should drop to **`0.0 GiB`** on `UD-IQ4_XS`).
- Decode tok/s (target **16–24 tok/s** at 8K/32K, **10–14 tok/s** at 128K).

*(Optional extra decode test)*: test `--strata-args ... --skip-weight 0.03` on coding prompts to see if skipping tiny
cold CPU experts adds +15–30% decode with no quality drop.

**Result — measured, and the premise is half wrong.**

`UD-IQ4_XS` **cannot load**, so the model half of Step 2 is blocked:

```
strata-poly: the GGUF has no tensor blk.6.loop_msg_query.weight
```

The file on disk is the pinned 15,733,755,232 B `unsloth/Inkling-2-Mini-GGUF` quant, but it is an *older* conversion: its
per-block loop tensors are named `loop_attn_{down,forget,gain}`, `loop_k_*`, `loop_v_*`, `loop_mlp_*` (12 a block), where
`UD-Q4_K_XL` has `loop_{q,k,v,ffn}_ctrl_*` (12) plus `loop_msg_query` (13), and IQ4_XS has **no `msg` tensor at all**
(734 vs 744 tensors, and the 146 names that differ are exactly those). Without `loop_msg_query` the looped core cannot compute
its messages, so this is a converter difference, not a flag: IQ4_XS needs either a re-quantization from the Q4_K_XL source or
`Inkling2` taught the older name scheme. Not attempted — it is a correctness change, not a benchmark.

The flag half **was** measured, on `UD-Q4_K_XL` (same model as the baseline, so the delta is the flags alone),
`--kv q8_0 --ram-experts-mib 10240`, one fresh server a row:

| Flags | Prompt | Prefill | Decode | VRAM | Experts |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (`6,640 MiB` peak) | 8,198 | 165.0 | 10.1 | 6,640 MiB | 791 |
| `--vram-reserve-mib 256` | 8,198 | **168.4** (+2.1%) | **10.1** (±0) | 6,896 MiB | **951** |
| `--vram-reserve-mib 256` | 32,015 | **97.4** (+2.6%) | **9.1** (±0) | 6,896 MiB | **951** |
| `--vram-pool --vram-reserve-mib 256` | 8,198 | **fails at 4,096 tokens** | — | — | 1,002 |

So the **residency prediction is right** (791 → 951, +160 experts) and the **speed prediction is wrong**: +2% prefill and
*no* decode change at either 8K or 32K. With a 10,240 MiB pinned RAM pool the experts come from RAM fast enough that
more of them on the card buys nothing — the pool, not residency, is the decode path. Step 2 is therefore not the "biggest
immediate decode win"; it is a ~2% prefill win and 256 MiB of VRAM.

`--vram-pool` is **not usable here**: it claims 15,770 MiB of virtual VRAM and leaves the lazy KV arena nothing, so the
first 4,096-token call dies with `no VRAM for the attention cache (2 MiB more)`. Do not pass it with a lazily growing KV.

### Step 3 — Optional (Cap at 30 min): Fixing the `blk 6` Fault in `read_layer_major()`
Because `read_layer_major()` only measured **+9–13% prefill** (1.09–1.13×), do **not** spend more than ~30 minutes
here — Step 4 (Tensor-Core MMA prefill) is the 5–10× prefill win. If you do fix it:
1. **Root cause**: `blk 0..5` is clean (`0.99997`), so the fault is at `blk 6` when `n_calls >= 2` (`c = 0` at `pos 0`,
   `c = 1` at `pos 1024`) share scalar `n_bank_` / `n_prev_bank_` and device buffers `bank_` / `prev_bank_` / `prev_out_`.
   - At `step 6` (`blk 6`, `block_start = true`), `c = 0` increments `n_bank_` from 1→2 before `c = 1` runs, so `c = 1`
     reads uninitialized `bank_[1]` and pushes into `bank_[2]`.
   - At `step 17` (`blk 17` pass 0 tail), `c = 1` overwrites `prev_bank_` / `prev_out_` before `c = 0` reads them at
     `step 18` (`blk 6` pass 1).
2. **Exact fix**: Store `(n_bank_, n_prev_bank_, bank_, prev_bank_, prev_out_)` in a per-call host struct `call_state[c]`:
   - Before `layer(il, call.m)` for call `c`, restore `n_bank_`, `n_prev_bank_`, and async-copy `call_state[c]` buffers
     to device (`bank_`, `prev_bank_`, `prev_out_`).
   - After `layer(il, call.m)` for call `c`, save `n_bank_`, `n_prev_bank_`, and async-copy modified buffers back to
     `call_state[c]`.
3. **Verify**: Dump `l_out` at `blk 6` (passes 0, 1, 2) and `blk 22` on a 2,048-token prompt; cosine vs call-at-a-time
   must be **`>= 0.99999`** on both `pos 0` and `pos 1024`, and greedy text must match character-for-character.

**Result — done, inside the cap, and the premise was wrong twice over (`4a39c701` on `inkling2-layer-major-wip`).**
The fault was **not** the shared `n_bank_` / `prev_bank_` state the guess above describes. It was two sizing errors:
1. `read_rows_` was clamped to `kReadCallSlots * rows_` (4,096), but every per-token scratch buffer in `assign_scratch()`
   is sized for `rows_`, and `bank_`'s stride is `rows_ * n_embd_`. A 4,096-row call therefore wrote past them.
2. Even sized correctly, the two reads chunk differently: layer-major read 8,192 as **8 calls of 1,024** where call-at-a-time
   took **one call of 8,192**. The conv history and the window ring differ at 1 ulp between chunkings, and the looped core
   (3 passes over `[6,18)`) amplifies that to a different token.

With `read_rows_ = rows_` on both paths the two reads are **byte-identical** over 128 generated tokens at 8,192
(`bench/lm-fix8k-b`, 163.6 vs 161.4 tok/s prefill, decode 9.7 vs 9.8), and layer-major is **1.01×**. The 1.09–1.13×
that motivated this step was an artifact of the 4,096-row calls the buffers could not hold — those calls were faster
precisely because they were wrong. **Conclusion: the layer-major path is correct and worth 1%, so it stays off**
(`read_layer_major()` returns `false`; the attempt, `tools/layer_major_ab.py`, and `tools/dump_diff.py` are on
`inkling2-layer-major-wip`). This closes Step 3.

### Step 4 — Tensor-Core MMA Prefill Attention (Biggest Prefill Win: 5–10× at 32K–128K)
Right now, `ink_attend_rel` / `ink_attend_rel_q8` uses scalar FP32 FMA for `Q·K` and `P·V`, which is why 128K prefill
takes 61.3 minutes. Replacing it with an MMA tensor-core kernel for `rows >= 32` removes the quadratic wall:
1. **Reference file**: Copy the warp-level MMA tile scaffolding from `src/poly/fattn_mma.cuh` (already in this repo
   and compiled for `sm_120`).
2. **Tile layout** (`dim = 128`, `Br = 64` query rows, `Bc = 64` key rows per tile):
   - Load `Q` tile `[64, 128]` into FP16 shared memory once per block.
   - For each key tile `j0` (step `Bc = 64`):
     a. Cooperatively load & dequantize `K` tile `[64, 128]` (from FP16 or Q8_0 codes × scales) into FP16 shared memory
        (`16 KB`).
     b. Compute FP32 `S = Q · K^T` (`[64, 64]`) using `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
     c. In registers on the `S` fragment: multiply by `scale * tau[row]`, add `kqb[(row, h, min(dist, n_keys-1))]`,
        and mask out-of-window / future keys (`j < lo || j > hi`) to `-INFINITY`.
     d. Run online softmax per row (starting from sink `m = 0, sum = 1`), rescale `O` accumulator, convert `P` to FP16.
     e. Cooperatively load & dequantize `V` tile `[64, 128]` into FP16 shared memory (`16 KB`), then accumulate
        `O += P · V` via MMA.
   - Normalize `O /= sum` and store to `out`.
3. **Dispatch**: Use this MMA kernel in `ink_attend_rel` / `ink_attend_rel_q8` when `rows >= 32 && dim == 128`; keep
   Split-K / warp-level kernel for `rows < 32` (decode).
4. **Verify**: Add an `attend mma` check in `src/poly/inkling2_test.cu` comparing against FP64 reference at `rows = 128`,
   `pos = 2048` (`max abs diff < 2e-3`), then re-measure 32K and 128K prefill (target: 32K from 95 → **400+ tok/s**,
   128K from 61 min → **~8–15 min**).

**Result — done, and the 32K target in point 4 was wrong (`e5b1f0a3`).** The kernel is
`ink_attend_rel_mma_kernel<Q8>` in `src/poly/inkling2_kernels.cu`: one block takes a head and 64 consecutive rows of it, a
warp 16 of those rows, and walks the keys 64 at a time with the online softmax in registers, in log2 units. It is a port of
`attn_flash128_kernel` (kernels.cu), which already had the tile scaffolding for `dim = 128`, a window, a ring and a per-head
sink, and it keeps `ink_attend_rel_kernel`'s semantics exactly: the sink as `m = 0, l = 1`, `scale * tau[r]` folded into the
query, the bias added in log2 units, `exp2f` for the weights, and the same window/future mask. `INK_MMA_SMEM` is 69,632 B
(two buffers × keys × values), and `STRATA_INK_MMA=0` forces the scalar path so that both can be measured on one binary.

Two faults had to be fixed before it was correct, and neither was in the arithmetic:

- **The key range came from the warp's rows, not the block's.** `hi = pos + r0 + nr` with `r0` the warp's first row makes a
  warp whose rows run past `rows` (the last block of a 1,024-row chunk is short when `rows % 64 != 0`) fetch a key range one
  block past the last valid key. The 64-row unit test did not catch it because `cudaMalloc` padding absorbed the overread; in the
  engine that slice sits inside a shared arena, so it was an illegal memory access at 3,072 tokens. Both `hi` and `lo` are now
  computed from the block (`nb = min(64, rows - row0)`).
- **The `attend q8` check was comparing two different quantizations.** Its reference was the scalar kernel, which quantizes the
  query to int8, while the MMA path keeps it in fp32 and only rounds it to fp16 — so the 2.989e-02 it reported was the scalar
  kernel's own error, and the MMA path is the more accurate of the two. The gate is now an FP64 reference at the engine's own shape
  (24 heads over 8 kv heads, 200 rows, `pos = 128`, `n_keys = 1025`), which is what point 4 asked for in the first place.

Verification, one fresh server per row, `--kv q8_0 --ram-experts-mib 10240`:

| Prompt tokens | Prefill before | Prefill after | Gain | TTFT before | TTFT after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,198 | 165.0 | **346.9** | **2.10×** | 49.7 s | 23.6 s |
| 32,015 | 94.9 | **328.3** | **3.46×** | 337.3 s | 97.5 s |
| 130,715 | 35.5 | **247.4** | **6.97×** | 3,678.7 s (61.3 min) | 528.3 s (8.8 min) |

The 8K cell was re-measured on the final binary after a linkage-only cleanup and gave 345.4 tok/s and 23.7 s,
the same within 0.4%; the table keeps the first figure and the committed `bench/mma-gate/8k` cell is the repeat.

`STRATA_INK_MMA=0` on the same binary reproduces 165.0 tok/s and 49.7 s at 8K exactly, so the gain is the kernel and
not the rebuild. VRAM stays 6,640 MiB and RSS 11.3 GiB at every length; output is coherent at 8.2K, 32K and 130.7K. The 23
kernel checks pass, including `attend ab` (5.631e-03 of 3.633) and `attend q8 ab` (5.732e-03 of 3.636) against FP64 at the
engine's shape.

The 128K target in point 4 (~8–15 min) was met at 8.8 min, and the 32K one (**400+ tok/s**) was not: the measured 328.3 is
3.46×, not 4.2×. That target came from an analytic fit of the three pre-kernel points that put attention at 25% of an 8K prefill,
57% of a 32K one and 85% of a 128K one; the fit was wrong, because removing attention entirely would have left 37.5 s at 8K and
the measurement is 23.6 s. Attention was therefore a larger share of prefill than the fit said at every length, and the 128K
quadratic wall is now gone rather than merely shortened: 247.4 tok/s at 130.7K is 7× the old figure and within 1.35× of the 32K one.


### Step 5 — Pass-Aware CUDA Graphs (`L.fixed = true`) + Graph Invalidation (+20–30% Decode)
1. **Close defect §8.3 first** (`src/poly/engine.cpp`):
   - In `shrink_cache()` (`~1409`) and `resize_card_cache()` (`~2118`), invalidate CUDA graphs (destroy `cudaGraphExec_t`
     in `passes_`, `tail_`, and every `Layer`'s `heads`, `tail`, `shared`, `card_moe`, `rows_heads` — mirroring the
     existing Metal invalidation at `~2124-2128`) whenever expert slots move.
2. **Add `tail_key()` virtual** (`src/poly/engine.hpp` + `src/poly/engine.cpp`):
   - `virtual int tail_key() const { return 0; }` (default 0 leaves all other models byte-identical).
   - Where `L.tail`, `L.shared`, and `L.card_moe` are indexed by `rec`, index by `(size_t) tail_key() * recs + rec`.
3. **Enable on `Inkling2`** (`src/poly/inkling2.hpp` + `src/poly/inkling2.cpp`):
   - Override `int tail_key() const override { return steps_[(size_t) cur_step_].pass; }`
   - In `head_key(il, n)`, return `steps_[(size_t) cur_step_].pass * (max_room_key + 1) + room_key(n)` on global layers
     (and `pass` on SWA layers).
   - Set `L.fixed = true` in `Inkling2::load_layer()`.
4. **Verify**: Run `tools/kv_agree.py` (with `--kv q8_0` and graphs on vs `--no-graphs`) and confirm exact match +
   re-measure decode tok/s at 8K / 32K / 128K.

### Step 6 — Agentic Coding Correctness Gates (Before Daily-Driving)
1. **End-to-end cosine gate on `UD-Q4_K_XL`**: `UD-Q4_K_XL` has **zero `Q4_K` tensors** (only `NVFP4`, `Q8_0`, `BF16`,
   `F32`), so `STRATA_DUMP_BIN` + `tools/dump_cos.py` vs `tools/ref_core.py` works today — run it and confirm
   cosine `>= 0.999` at `blk.6`, `blk.23`, and final logits.
2. **Tokenizer byte-for-byte parity**: Compare `tools/strata_tokenizer.py` on 20 mixed coding strings (Python/C++/TS,
   JSON, tabs, Unicode) against Hugging Face `transformers` (`AutoTokenizer.from_pretrained("thinkingmachines/Inkling-2-Mini")`).
3. **Tool-call & reasoning tests**: Add unit tests in `serve/test_models.py` verifying `<|content_thinking|>` →
   `reasoning_content` and `<|content_invoke_tool_json|>` → OpenAI `tool_calls`.
4. **Daily-drive server config**: Start `serve/server.py` with `--kv q8_0 --vram-reserve-mib 256 --vram-pool --ram-experts-mib 10240 --expert-profile bench/inkling2-code.profile --pinned-session` so multi-turn coding sessions reuse the cached KV prefix with 1–5 s TTFT.

### Step 7 — DSpark Speculative Decoding (Biggest Decode Multiplier: ~2.0–2.8×)
Currently `Inkling2` runs with `drafts() == false` (1 token per 48-layer pass). Because Inkling 2 Mini is an offloaded
MoE, speculative verification (`Engine::window(tokens, t)` with `t = 4`) coalesces expert reads across all 4 tokens
(`engine.cpp:4066`) and amortizes all non-expert + KV reads across 4 tokens at once.

- **Pinned drafter files** (`unsloth/Inkling-2-Mini-GGUF` @ `ea0c79b2ecd73669e774c8519b79bfabfddbc027`):
  - `dspark-Inkling-2-Mini-Q8_0.gguf` — **402,278,560 bytes (402 MB)**, SHA-256
    `2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773` (fits 100% in VRAM; ~2–3 ms/draft; losing ~85
    cached experts costs ~5% per pass while gaining +100–180% effective decode tok/s).
  - `dspark/dspark-Inkling-2-Mini-BF16.gguf` — 648 MB (lossless reference).
- **Architecture (`mtp_config` from `thinkingmachines/Inkling-2-Mini/config.json`)**:
  - 3 dense RoPE + SwiGLU layers (`hidden_size = 2304`, `intermediate_size = 9216`, `num_attention_heads = 24`,
    `num_key_value_heads = 8`, `head_dim = 128`, `rope_theta = 1000000.0`, `rms_norm_eps = 1e-6`).
  - Attention split: `local_layer_ids = [0, 2]` (`sliding_window_size = 1024`), layer `1` is global.
  - Markov/RNN head: `block_size = 8`, `markov_rank = 256`, `markov_head_type = "rnn_hidden"`, tapping target hidden
    states at logical steps **`target_layer_ids = [5, 23, 41, 48]`**.
- **How to wire into `strata-poly`** (reference `ggml-org/llama.cpp#32218` for exact DSpark tensor names & math):
  1. Inspect `dspark-Inkling-2-Mini-Q8_0.gguf` headers with `tools/gguf_reader.py` to confirm all tensor names.
  2. Pass `--draft-model <path-to-dspark-Inkling-2-Mini-Q8_0.gguf>` (already parsed into `opt_.draft_model` / `dg_` in
     `Engine::load()`, `src/poly/engine.cpp:2678`).
  3. In `Inkling2`: capture the 4 target hidden states at logical steps `[5, 23, 41, 48]` into a small scratch buffer
     (`4 × rows_ × 2304` floats) during the main pass.
  4. Load the 3 dense layers + Markov head from `dg_` into VRAM and override `Inkling2::draft_step(int32_t token, int i)`
     (`src/poly/engine.hpp:418`) to produce draft `logits_` (`Engine::draft()`, `window()`, `keep()`, and `snap_`
     recurrent-state rollback handle the rest automatically).
  5. **Verify**: Run greedy decoding (`--temp 0`) with `--draft-model` (`--window 4`) vs `--no-draft` and confirm
     **byte-identical token output**, then measure acceptance rate and effective decode tok/s at 8K / 32K / 128K.

#### Step 7 — what the reference actually says, and why this is a port and not a flag

Recovered from `ggml-org/llama.cpp` master `src/models/dspark.cpp`, `common/speculative.cpp` (PR #32218), and
`conversion/inkling2.py` (`Inkling2DSparkModel`). The GGUF's own metadata is `general.architecture = dflash` with
`dflash.*` keys, so `Inkling2` must learn a **second architecture**, not reuse `Engine`'s draft block.

**Block diffusion, not autoregression.** The drafter decodes a whole trained block of `block_size = 8` positions in
**one non-causal pass** (`dflash.attention.causal = False`), then the verifier checks up to `n_max` of them:
- slot 0 is the anchor (the last committed token, `dp.id_last`); slots 1..7 are `mask_token_id` (0);
- `dflash.sample_from_anchor = False` here, so slot 0 is a bonus token and **7** tokens are drafted (`i_draft_beg = 1`);
- `dflash.has_confidence_head = False`, so there is no `p_min` truncation; `dflash.has_markov_context` is **absent**,
  which selects `nextn_masked = true` and the simple `n_block_tokens = n_draft + 1` path;
- per position the graph is fed an **injected state vector of width `n_embd` (2304)**, not a token embedding:
  slot 0 gets the target's hidden state for the committed token, the others get `mask_embd` (greedy) or
  `[u, 1/temp, 0…]` (probabilistic). There is **no `token_embd` table** in this GGUF at all — confirmed: 54 tensors,
  no embedding matrix.

**Layer stack** (3 layers, `n_embd = 2304`, `n_ff = 9216`, 24 heads / 8 KV heads, `head_dim = 128`,
`rope_theta = 1e6`, `rms_eps = 1e-6`, `sliding_window_pattern = [True, False, True]` → layers 0 and 2 window at 1024,
layer 1 global):

```
x  = injected_state (+ mask_embd on the mask slots)
for l in 0..2:
    n   = rms_norm(x, attn_norm[l])
    q   = rms_norm(reshape(n @ attn_q[l]),  attn_q_norm[l])   # 2304 -> 3072, 24 x 128
    k   = rms_norm(reshape(n @ attn_k[l]),  attn_k_norm[l])   # 2304 -> 1024,  8 x 128
    v   = n @ attn_v[l]                                       # 2304 -> 1024
    kc  = rms_norm(reshape(ctx @ attn_k_ctx[l]), attn_k_ctx_norm[l])   # the DFlash inject path
    vc  = ctx @ attn_v_ctx[l]                                          # from the fused target state
    q,k = rope(q, k, pos)                        # non-causal over the block; K/V = [context | block]
    x  += attn_output[l] @ attend(q, [kc|k], [vc|v])
    x  += ffn_down[l] @ silu(ffn_gate[l] @ rms_norm(x, ffn_norm[l]) * ffn_up[l] @ …)
hidden = rms_norm(x, output_norm)               # 2304 per position
```

**Markov head** (`rnn_hidden`, rank `R = 256`) — this is where this variant diverges from the DSV4 code:
```
base   = unembed(hidden)                         # the target's lm_head, borrowed; 201024 logits
state  = 0
for i in 1..7:
    w1_prev = markov_w1[token[i-1]]              # row of the folded matrix, [R]
    joint   = joint_proj @ [state | w1_prev | hidden[i] | ctx_fused]  + joint_proj.bias   # [3R]
    gate    = sigmoid(joint[0:R]); cand = tanh(joint[R:2R])
    state   = gate*state + (1-gate)*cand
    logits[i] = base[i] + markov_w2 @ tanh(joint[2R:3R])
```
`markov_w1` in the GGUF is already **folded**: `conversion/inkling2.py::_fold_markov_chunk` computes
`unembed[rows] @ dspark_hidden_out.weight`, so `markov_w1 [256, 201024]` is `(unembed ∘ hidden_out)` and needs no
separate 2304-wide head. `markov_w2` is `[256, n_vocab_draft]` in llama.cpp but this GGUF has only
`dspark_hidden_out` + `dspark_norm`, so the `w2` bias term is the unfolded path here.

**The discrepancy to resolve before writing code.** llama.cpp declares
`dspark_joint_proj = {n_embd + 2*rank, 3*rank}` = `[2816, 768]`, and its `joint` is
`concat(concat(state, w1_prev), hidden)` = `256 + 256 + 2304`. This drafter's tensor is **`[5120, 768]`**, and
`5120 - 512 = 4608 = 2 × 2304`. So the Inkling 2 variant conditions the Markov chain on **two** 2304-wide vectors —
the drafter's own block hidden *and* the target's fused hidden (`dspark_fuse [4,4]` over the four tapped layers) —
which the DSV4/Mistral code does not do. That is the one inference in this spec that must be validated first, and the
cheapest way is a two-position check: force `joint`'s last 2304 to the target state and compare against the BF16
reference drafter's own logits.

**Integration points that already exist** (so the port is the forward pass, not the plumbing): `opt_.draft_model` /
`dg_` (`engine.cpp:2678`), `drafts()`, `guesses_at_once()` (`engine.hpp:313-315`, the block-at-once path DSpark wants),
`draft_step()` (`engine.hpp:418`), and `window()` / `keep()` / `snap_` which already roll back a rejected block.
