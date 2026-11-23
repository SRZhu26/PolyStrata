# Inkling 2 Mini 26B-A5B (`inkling2_loop`)

Thinking Machines Lab's [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) in
[Unsloth's `Mainline_Support` GGUFs](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support)
uses the `inkling2_loop` architecture (renamed `inkling2` from the Mainline_Support release on).

> **Status:** Implemented and measured on one machine — the RTX 5060 Laptop (8 GB VRAM, 32 GB RAM, Windows,
> CUDA 13.1, `sm_120`). It loads, prefills, decodes and produces coherent text at short and medium context.
> It is **not** production-ready: long-context output is intermittently corrupted (see
> [Known defects](#known-defects)), and three throughput capabilities are deliberately disabled
> (see [Disabled capability hooks](#disabled-capability-hooks)). Only the CUDA backend has ever been built.
> For the tensor inventory and the phased plan, see **[inkling2_plan.md](inkling2_plan.md)**.

## Architecture and KV cache math

From `Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file` (`ea0c79b2ecd73669e774c8519b79bfabfddbc027`):

- **Looped core**: 24 physical layers (`0..23`), with layers `6..17` run 3× using shared weights and rank-64 loop
  gates → **48 logical layer applications per token** (26.2B total, 3.4B unique active, 5.4B effective active).
- **Attention split**: 20 physical local SWA layers (`sliding_window = 1024`) and 4 global layers (`5, 11, 17, 23`)
  → **40 local SWA + 8 global** logical KV caches per sequence (`n_head = 24`, `n_kv = 8`, `head_dim = 128`).
- **Position**: `rope.dimension_count = 0` — there is no RoPE. Position enters only through a `relative_length = 16`
  bias table with `relative_extent = 1024`, `log_scale_floor = 8192`, `log_scale_alpha = 0.1`, `qk_clip = 22.627`
  and `relative_clip = 8.0`.
- **Per-token KV size** (using `strata-poly`'s Q8_0 block layout from [Q8_KV.md](../Q8_KV.md)):
  - **FP16**: `4,096 B/tok/layer` → **32,768 B/tok** (32 KiB/tok) across 8 global layers + **320 MiB** SWA rings
    (`1024 + 1024` prompt-batch rows).
  - **Q8_0**: `2,176 B/tok/layer` (`1,088 B` per K or V row) → **17,408 B/tok** (17 KiB/tok) across 8 global layers
    + **170 MiB** SWA rings (46.88% less KV VRAM).
- **Tensor sizes (from GGUF headers)**:
  - `UD-IQ4_XS` (15.73 GB, provisional default, KLD 0.0109): 14.04 GB routed experts, 1.69 GB non-expert weights, 0 non-NVFP4 `.scale` layers.
  - `UD-Q3_K_XL` (13.63 GB, speed/low-RAM option, KLD 0.0174): 12.20 GB routed experts, 1.42 GB non-expert weights, 8 non-NVFP4 `.scale` layers.
  - `UD-IQ3_XXS` (11.66 GB, smallest Q3, KLD 0.0269): 10.30 GB routed experts, 1.35 GB non-expert weights, 20 non-NVFP4 `.scale` layers.
  - `UD-Q4_K_XL` (17.63 GB, bringup reference, KLD 0.0107): 15.80 GB routed experts (`NVFP4`, `Q8_0` only), 1.83 GB non-expert weights.
  - `UD-Q4_K_M` (16.61 GB, KLD 0.0108): 14.78 GB routed experts (`NVFP4`, `Q5_K`, `Q8_0`), 1.83 GB non-expert weights.

### Estimated VRAM budget on 8 GB VRAM / 32 GB RAM (RTX 5060 Laptop)

With ~7.0 GiB usable VRAM on Windows (~1.71 GiB non-expert weights + ~0.9 GiB scratch/overhead leaves ~4.4 GiB
for KV + hot experts):

| Context | FP16 KV Total | Q8_0 KV Total | Est. VRAM Left for Experts (Q8_0) |
| --- | ---: | ---: | ---: |
| **32K (default)** | 1.31 GiB | **0.70 GiB** | **~3.7 GiB** |
| 64K | 2.31 GiB | 1.23 GiB | ~3.2 GiB |
| 128K | 4.31 GiB | 2.29 GiB | ~2.1 GiB |
| 192K | 6.31 GiB | 3.35 GiB | ~1.0 GiB |
| 256K | 8.31 GiB | 4.42 GiB | ~0 GiB |

`setup.py --family inkling2` defaults to **32K** context (128K available via `--context 131072`), **`--kv q8_0`**,
and **`UD-IQ4_XS`** (provisional default; `UD-Q3_K_XL`, `UD-IQ3_XXS`, `UD-Q4_K_XL`, and `UD-Q4_K_M` are also
offered, with the final default chosen after Stage 4 benchmarks).

## Implementation

The architecture is not reachable through the generic engine path, so it has its own implementation:

| File | Lines | Contents |
| --- | ---: | --- |
| `src/poly/inkling2.cpp` | 758 | the `Inkling2` class: 29 hook overrides |
| `src/poly/inkling2.hpp` | 170 | its interface and capability flags |
| `src/poly/inkling2_kernels.cu` | 585 | 12 CUDA kernels |
| `src/poly/inkling2_test.cu` | 485 | kernel tests against an FP64 reference |

Engine-side integration (`src/poly/engine.cpp`, `engine.hpp`, `kernels.hpp`, `cpu_experts.cpp/hpp`,
`CMakeLists.txt`, `tools/strata_tokenizer.py`) adds the arch registration, the `--kv q8_0` allowance, the shared
`gate_up` NVFP4 scale, the `situ` activation hook, and the CPU-expert `situ` path.

Kernels with no equivalent anywhere else in the engine:

- `ink_situ` / `ink_situ_rows` — `situ` replaces SwiGLU/GeGLU (`beta = 4.0`, `linear_beta = 25.0`).
- `ink_rel_table` + `ink_attend_rel` — the relative-position bias table and the attention that consumes it.
- `ink_depth_mix` / `ink_bank_states` — the depth-attention bank.
- `ink_conv` / `ink_conv_state` — the convolutional recurrent state (`k = 4`).
- `ink_tau` — position-dependent temperature, flat at `1.0` below `log_scale_floor = 8192`.
- `ink_clip_l2` — the `qk_clip = 22.627` per-head L2 clip.
- `ink_control` — the loop gate's `down`/`forget`/`gain` state update.
- `ink_finish_logits` — softcap `30.0` and stripping the 966 rows of vocabulary padding.

KV caches hang off the **visits**, not the blocks, because a core layer is visited three times over the same
weights: window layers get a `ring = min(max_context, window + rows)` ring from `seq_alloc`, full-context layers get a
row at a time from the growing `kv_alloc` arena.

## Reference and verification

**There is no `llama.cpp` baseline for this architecture.** Build `b11669` (commit `620d2225c`) refuses the file with
`unknown model architecture: 'inkling2_loop'`, and neither `llama-server.exe` nor `llama.dll` contains the string. Mainline
`llama.cpp` has no `inkling2` implementation, so a `llama.cpp` speed or parity comparison is not merely unmeasured — it is
undefined. The reference is the FP64 oracle in `tools/ref_core.py`, which runs the whole model in double precision from the
same GGUF.

- `tools/ref_prelude.py` — validated per-stage oracle, used by `tools/dump_cos.py`.
- `tools/ref_core.py` — the full model: all 74 stages plus logits run end to end.
- **Kernel parity:** `poly_inkling2_test` compares the kernels against the FP64 reference of the same graph; its 19
  checks pass, including the split-K flash-decoding paths (`attend split` 3.3e-07 of 0.449, `attend q8 spl` 2.1e-07 of 0.423).
- **End-to-end cosine gate:** **passes.** The `STRATA_DUMP_BIN` + `tools/dump_cos.py` comparison against `ref_core.py` on
  `UD-Q4_K_XL` reads `blk.6` 0.999988758, `blk.23` 0.999988499 and the logits 0.999763105, against a 0.999 gate; all 24
  layers are at or above 0.99997 (`blk.0` 0.999978682 … `blk.16` 0.999971239). `UD-Q4_K_XL` holds no `Q4_K` tensors
  (`Q8_0` 154, `F32` 518, `BF16` 32, `NVFP4` 40), so the gate never needed the NVFP4 source checkpoint.

## Measured results (RTX 5060 Laptop, `UD-Q4_K_XL` 17.6 GB, FP16 KV, streaming experts)

> **These are the bring-up figures, before the tensor-core prefill kernel and before `--prompt-batch 2048`.** Current, on the
> same machine with `--kv q8_0`: **prefill 345.4 / 328.3 / 247.4 tok/s at 8,198 / 32,015 / 130,715 tokens** and decode
> 10.0 / 9.7 / 6.3 tok/s (progress §4.6, `0b1d7dd4`), rising to **378.2 at 8K and 389.3 at 32K** with `--prompt-batch 2048`
> (progress §4.7). The table below is kept as the pre-kernel baseline it is; [inkling2_plan.md](inkling2_plan.md) §4b has the
> measured decomposition of what is left and the ranked list of what has never been tried.

| Prompt | Prefill tok/s | Decode tok/s | VRAM peak | RSS peak |
| --- | ---: | ---: | ---: | ---: |
| 8,198 | 71.7 | 5.0 | 7,498 MiB | 11.53 GiB |
| 16,343 | 84.4 | 10.1 | 6,640 MiB | 12.54 GiB |
| 130,715 | 23.4 (average) | — | — | — |

Prefill curve over the 128K prompt: 111.8 tok/s at 1,024 tokens, 68.3 at 8,192, 39.4 at 32,768, 23.3 at 65,536,
16.8 at 98,304, 13.7 at 122,880. Model load takes 12.1 s. The 8K decode figure comes from a run that degenerated
into a loop, so it understates throughput. Peak system memory was 25.2 GiB with 0.95 GiB of swap.

Gates: VRAM ≤ 7.5 GiB — passes, but only by 182 MiB at 8K. RAM ≤ 26 GiB — passes at 25.2 GiB, also narrowly.
Decode ≥ 1.5× `llama.cpp` — **cannot be evaluated**, because no `llama.cpp` result exists for this architecture.

## Disabled capability hooks

Three hooks in `inkling2.hpp` are switched off, each marked `// until Phase N`. All three cost throughput, not
correctness:

| Hook | Value | Effect |
| --- | --- | --- |
| `read_layer_major()` | `false` | A long prompt cannot be read layer-major across calls, so experts are **re-read from disk for every chunk** instead of being shared. This is the largest prefill cost at long context. |
| `halves_ok()` | `false` | A window's tokens cannot be pipelined through the layers in two halves. |
| `state_whole()` | `true` | The recurrent state is saved whole rather than a layer at a time, because a core layer's index does not name its visit's state. |

The measured numbers above are therefore a **lower bound** on this engine's throughput, not a ceiling.

## Known defects

1. **Intermittent long-context corruption.** At roughly 10,900 tokens of prefill, about two runs in three degenerate
   into a repetition loop; the same prompt, same binary and a fresh server sometimes produce coherent text (10,890 tokens
   coherent vs 10,898 degenerate — a race, not a length threshold). Ruled out: sampling, `tau`, softcap, vocabulary
   padding, the SWA ring, `cover()` growth, `shrink_cache`, swap prefetch, and the attention kernel itself (re-running a
   cached prefix over the same KV is always coherent). Remaining suspects are the `kStageRows = 32` staged expert launches
   in `moe_rows()` and the per-chunk recurrent state copies in `copy_seq()`.
2. **Sixth sequential request on one server dies** with `the depth attention's bank is full (5 of 5 at step 42)`.
   `max_states_ = 5` and a single read should peak at 4, so one bank state leaks per request.
3. **Latent CUDA-graph invalidation gap.** `resize_card_cache()` and `shrink_cache()` move experts between slots, but
   CUDA graphs are invalidated only under `#if defined(STRATA_GPU_METAL)`, and the pass-graph key omits the cache layout.
   Harmless today only because `L.card == 1` is false for streamed experts.

## Limitations

These configurations are **not supported**. They are recorded so the gap is explicit rather than discovered by a user.

| Configuration | Status | Why |
| --- | --- | --- |
| **Full GPU residency** (all 17.6 GB in VRAM) | **Untested** | No machine with ≥ 18–20 GB of VRAM has run it. The code path should work — experts are streamed by policy, not by necessity — but it has never been executed. Treat as unverified, not as supported. |
| **HIP / ROCm (AMD)** | **Unsupported** | `inkling2_kernels.cu` hardcodes `cudaStream_t` at line 28 with no `__HIP__` branch. Sibling `kernels.cu` does have one; these kernels do not, so an AMD build fails to compile. |
| **Metal / macOS (Apple Silicon)** | **Unsupported** | There is no Metal variant of the kernels. `inkling2_kernels.cu` is the only implementation, and it is CUDA. |
| **CPU-only (no GPU)** | **Unsupported** | `strata-poly` is only built under `CUDA OR HIP`. Attention, the short convolution, depth mixing and the loop control are card kernels with no host fallback. |
| **Multi-GPU layer split** (`--layer-split`) | **Untested** | The flag exists, but per-visit KV caches plus `state_whole()` make it riskier than for other models. |
| **CPU experts** (`--ram-experts-mib`) | **Supported** | `cpu_experts()` carries the `situ` path, so CPU experts compute the same activation as `ink_situ_rows_kernel`. |

Full GPU residency is the most likely configuration to change the picture: at 8 GB the KV arena and expert streaming
compete for the same VRAM, which is exactly the pressure under which the corruption appears.
