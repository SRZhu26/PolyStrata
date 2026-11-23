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
