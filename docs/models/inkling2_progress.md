# Inkling 2 Mini 26B-A5B — Progress Report

Status of PolyStrata's `strata-poly` engine on **Thinking Machines Lab's Inkling 2 Mini** (`inkling2_loop`, renamed
`inkling2` from the Mainline_Support release on), measured on the **RTX 5060 Laptop — 8,151 MiB VRAM, 32 GB RAM,
Windows 11, CUDA 13.1, `sm_120`**.

Companion documents: [inkling2.md](inkling2.md) (architecture and measured results), [inkling2_plan.md](inkling2_plan.md)
(phased plan, now adjusted to what was actually found).

> **One-line verdict:** it loads, prefills, decodes and writes coherent text at short and medium context, and it is the
> only runtime that can execute this architecture at all — but long-context output is intermittently corrupted, and three
> throughput capabilities are deliberately switched off.

---

## 1. Branch and commit state

| Ref | Commit | Contents |
| --- | --- | --- |
| `main` | `8d812d5c` | untouched |
| `add-inkling2-mini` (PR head) | `59f21100` | untouched — the PR to `main` is unaffected by this work |
| `inkling2-engine-local` | `0432e9a6`, `76008937` | **this work**, local only, not pushed |

`origin` is `https://github.com/SRZhu26/PolyStrata.git`. The engine work was committed to `inkling2-engine-local`
precisely so it does not appear in the `add-inkling2-mini` → `main` pull request.

Left untracked on purpose: `bench/*` outputs, `.workbuddy/`, and a stray `--help` file produced by an accidental
redirect.

---

## 2. Environment and build

- Windows 11, Visual Studio Build Tools (MSVC), CMake, CUDA 13.1, target `sm_120` (Blackwell, RTX 5060 Laptop).
- Build: `tools/build_poly_cuda_local.sh` → `BUILD_OK`; binary `build-poly/strata-poly.exe` (88 MB).
- Model under test: `Inkling-2-Mini-UD-Q4_K_XL.gguf`, 17,629,760,736 B (17.6 GB / 16.42 GiB), 24 physical layers.
- KV cache: FP16 in every run below. `--kv q8_0` is wired and accepted for this architecture but was not the
  configuration measured.
- Tokenizer: `bench/tok-inkling2/tokenizer` (converted from the GGUF), used by the server.

---

## 3. What is implemented

The architecture is not reachable through the generic engine path, so it has its own implementation. Nothing here is a
stub: every kernel listed is exercised by the runs in §4.

| File | Lines | Contents |
| --- | ---: | --- |
| `src/poly/inkling2.cpp` | 758 | `Inkling2` class, 29 overrides |
| `src/poly/inkling2.hpp` | 170 | interface and capability flags |
| `src/poly/inkling2_kernels.cu` | 585 | 12 CUDA kernels |
| `src/poly/inkling2_test.cu` | 485 | kernel tests against an FP64 reference |

Engine-side integration: arch registration for `inkling2_loop` / `inkling2` in `Engine::open()`, the `--kv q8_0` allowance,
the shared `gate_up` NVFP4 scale across both halves, the `moe_act()` hook, `situ_` / `situ_beta_` / `situ_linear_beta_`
members, `state_whole()`, and the `situ` path in `cpu_experts()`.

Kernels with no equivalent anywhere else in the engine:

| Kernel | Replaces / provides |
| --- | --- |
| `ink_situ` / `ink_situ_rows` | `situ` activation (`beta = 4.0`, `linear_beta = 25.0`) instead of SwiGLU/GeGLU |
| `ink_rel_table` | the `relative_length = 16` bias table, `extent = 1024`, clip `8.0` |
| `ink_attend_rel` | attention consuming that bias — **there is no RoPE** (`rope.dimension_count = 0`) |
| `ink_clip_l2` | per-head `qk_clip = 22.627` |
| `ink_conv` / `ink_conv_state` | causal short convolution, `k = 4`, plus its recurrent state |
| `ink_depth_mix` / `ink_bank_states` | the depth-attention ("Attention Residuals") bank, `block_size = 6` |
| `ink_tau` | position-dependent temperature, flat at `1.0` below `log_scale_floor = 8192` |
| `ink_control` | the rank-64 loop gate's `down` / `forget` / `gain` state update |
| `ink_finish_logits` | softcap `30.0` and stripping the 966 rows of vocabulary padding |

**KV caches hang off the visits, not the blocks.** A core layer (`blk.6..17`) is visited three times over the same
weights, so its layer index does not name its KV. Window layers get a `ring = min(max_context, window + rows)` ring from
`seq_alloc`; the 8 full-context visits get a row at a time from the growing `kv_alloc` arena. This is why the KV arena
grows during a long prefill (`cover()` at `end=1` → `rows=4096`, `end=5120` → `rows=8192`, `end=9216` → `rows=12288`).

---

## 4. Measured results

All numbers: `UD-Q4_K_XL`, FP16 KV, experts streamed from disk, RTX 5060 Laptop.

| Prompt tokens | Prefill tok/s | Decode tok/s | VRAM peak | RSS peak |
| ---: | ---: | ---: | ---: | ---: |
| 8,198 | 71.7 | 5.0 | 7,498 MiB | 11.53 GiB |
| 16,343 | 84.4 | 10.1 | 6,640 MiB | 12.54 GiB |
| 130,715 | 23.4 (average) | — | — | — |

Model load: **12.1 s**. Peak system memory **25.2 GiB**, swap **0.95 GiB**.

Prefill over the 128K prompt, measured incrementally to 96% before the run was stopped:

| Tokens | Elapsed | Marginal tok/s |
| ---: | ---: | ---: |
| 1,024 | 10 s | 111.8 |
| 8,192 | 110 s | 68.3 |
| 32,768 | 599 s | 39.4 |
| 65,536 | 1,745 s | 23.3 |
| 98,304 | 3,454 s | 16.8 |
| 122,880 | 5,114 s | 13.7 |

A full 130,715-token prefill projects to ≈ 5,750 s (≈ 96 min). The fall-off sits between `1/√t` and linear, which is
the expected shape: a quadratic part from the 8 full-context visits and a linear part from the 40 window visits plus FFN.

The 8K decode figure of 5.0 tok/s comes from a run that degenerated into a loop, so it understates throughput. The 16K
figure of 10.1 tok/s is the only clean decode number.

---

## 5. Gates against the plan's criteria

| Gate | Criterion | Result |
| --- | --- | --- |
| Peak VRAM | ≤ 7.5 GiB (7,680 MiB) | **pass**, by only 182 MiB at 8K |
| Peak RAM | ≤ 26 GiB | **pass** at 25.2 GiB, narrowly |
| Decode vs `llama.cpp` | ≥ 1.5× same-day baseline | **not evaluable** — no baseline exists (§6) |
| Cosine per stage | ≥ 0.999 | **not yet run** (§7) |
| Kernel parity | vs FP64 | **pass** — `poly_inkling2_test`, 17 checks |
| Long-context needle | 100% recall at 32K/64K/128K | **not yet run**, and blocked by the corruption defect |
| Q8_0 KV | FP16 vs Q8_0 agreement ≥ 97% | **not yet run** |

---

## 6. There is no `llama.cpp` baseline, and there cannot be one

Build `b11669` (commit `620d2225c`) refuses the file:

```
llama_model_load: error loading model: unknown model architecture: 'inkling2_loop'
```

Neither `llama-server.exe` nor `llama.dll` contains the string `inkling`. Mainline `llama.cpp` has no `inkling2`
implementation, so the plan's "decode ≥ 1.5× `llama.cpp`" gate is not merely unmeasured — it is undefined. Executing it
would require porting the same ~2,000 lines, including the nine kernels in §3.

Note for the record: an earlier `bench/cmp8k` failure that looked like a `llama.cpp` limitation was in fact a bug in the
harness (`-ctk fp16`; `llama.cpp` spells it `f16`). The architecture refusal above is a separate, real result from
`bench/cmp8k-llama`.

The only reference that exists is the FP64 oracle we wrote:

- `tools/ref_prelude.py` — validated per-stage oracle, used by `tools/dump_cos.py`.
- `tools/ref_core.py` — the full model: all 74 stages plus logits, now runs end to end.
- `tools/ref_blk0.py`, `tools/ref_expert.py` — narrower oracles used during bringup.

The plan's Stage 1 references `tools/reference/inkling2.py` (ninfer-all's FP64 reference). **That file does not exist in this
repository**; `tools/ref_core.py` is the substitute.

---

## 7. Verification tooling state

| Tool | State |
| --- | --- |
| `poly_inkling2_test` | builds; 17 kernel-level checks against FP64 **pass** |
| `tools/ref_core.py` | runs end to end — all 74 stages plus logits |
| `tools/dump_cos.py` | exists; **the end-to-end cosine gate has not been run** |
| `tools/kv_agree.py` | exists; **not run** for this model |
| `tools/needle_bench.py` | exists; **not run** for this model |
| `tools/probe_context.py` | new; drives one server across increasing prefix lengths |
| `tools/strata_tokenizer.py` | `inkling2` mapped to a ported `GPT4O_PATTERN`; **byte-for-byte parity unverified** |

**Tokenizer parity is unverified too.** `strata_tokenizer.py` now maps `inkling2` to a `GPT4O_PATTERN` ported from
`llama.cpp`'s `LLAMA_VOCAB_PRE_TYPE_GPT4O`, with `clean_spaces = false` and no BOS. That is a port, not a measurement:
no byte-for-byte comparison against `llama-tokenize` or `transformers` was recorded. A tokenizer that counts differently
would invalidate the token counts in §4, so this should be checked before the §4 numbers are quoted externally.

**Kernel parity is not the end-to-end gate.** The 17 passing checks compare each kernel against an FP64 reference of the
same graph. They do not establish that the assembled 48-visit model matches `ref_core.py`; that requires
`STRATA_DUMP_BIN` plus `dump_cos.py` at `blk.6`, `blk.23` and the final logits, and it has not been run.

---

## 8. Known defects

### 8.1 Intermittent long-context corruption (blocking)

At roughly 10,900 tokens of prefill, about **two runs in three** degenerate into a repetition loop ("The lighter keeper is a
keeper of the light"). The same prompt, the same binary and a fresh server sometimes produce coherent text:
**10,890 tokens coherent vs 10,898 degenerate**. That pair proves a race rather than a length threshold.

It is not sampling: `--temp 0.7 --top_p 0.95` loops identically to greedy. It is not the depth-attention temperature
(`tau = 1.0` below 8192), not the softcap or vocabulary padding (both correct in `ink_finish_logits_kernel`), not the SWA
ring or window, and not `n_keys` (only the bias table's width, 1025; the sweep is driven by `pos`).

Ruled out by experiment:

| Suspect | How it was ruled out |
| --- | --- |
| Sampling / temperature / top_p | identical loop under temp 0.7 + top_p 0.95 |
| `tau` log-scaling | `tau = 1.0` below 8192 |
| Softcap, 966-row vocab padding | verified correct in `ink_finish_logits_kernel` |
| SWA ring and window | probe at successive prefix lengths |
| `cover()` KV growth | logs show correct growth in coherent runs too |
| `shrink_cache()` | fires in coherent runs (`cut=3712 MiB cached=770`) |
| Expert swap prefetch | `--expert-swaps 0` still fails, half of runs |
| The attention kernel | a cached `prompt_n=1` re-run over the same KV is always coherent |
| Slot publication ordering | `L.slot[e]` is set only after `h2d()` lands (`engine.cpp:2353-2356`, `2370-2373`) |

Remaining suspects, both only exercised by a long prefill: the `kStageRows = 32` staged expert launches in `moe_rows()`, and
the per-chunk recurrent state (`state_`, `bufs_` copies in `copy_seq()`). A 1-token continuation exercises neither.

**A blind spot made this harder than it needed to be:** `serve/server.py` opens the engine's stderr as `self.log`, which is
`subprocess.DEVNULL` unless the config carries a `"log"` key. Every probe written into `engine.cpp` was invisible until
`strata_config()` in `tools/bench_128k_local.py` was given one. An earlier conclusion that `cover()` never ran was drawn
from that silence and has been retracted.

### 8.2 Sixth sequential request on one server dies

```
the depth attention's bank is full (5 of 5 at step 42)
```

`max_states_ = 5`, and a single read should peak at 4, so one bank state leaks per request. The 6th request is the one
that fails.

### 8.3 Latent CUDA-graph invalidation gap

`resize_card_cache()` (`engine.cpp:2118`) and `shrink_cache()` (`engine.cpp:1409`) move experts between slots, but CUDA
graphs are invalidated only under `#if defined(STRATA_GPU_METAL)` (`2124-2128`), and the pass-graph key
(`engine.cpp:5042`) is `{n, head_key(...)}` with no cache layout in it. Harmless today only because `L.card == 1` is
false for streamed experts, which is the configuration every number above was measured in.

---

## 9. Disabled capability hooks

Three hooks in `inkling2.hpp` are switched off, each marked `// until Phase N`. All three cost throughput, not
correctness.

| Hook | Value | Consequence |
| --- | --- | --- |
| `read_layer_major()` | `false` | A long prompt is not read layer-major across calls, so experts are **re-read from disk for every chunk** instead of shared. Largest single prefill cost at long context. |
| `halves_ok()` | `false` | A window's tokens are not pipelined through the layers in two halves. |
| `state_whole()` | `true` | Recurrent state is saved whole rather than a layer at a time, because a core layer's index does not name its visit's state. |

Because `read_layer_major()` is off, **every number in §4 is a floor, not a ceiling.**

---

## 10. Configuration support

| Configuration | Status | Reason |
| --- | --- | --- |
| Streamed experts, 8 GB card (this laptop) | **Supported, measured** | every number in §4 |
| Full GPU residency (all 17.6 GB in VRAM) | **Untested** | needs ≈ 18–20 GB VRAM; no such machine has run it. Should work — experts are streamed by policy, not necessity — but treat as unverified, not supported |
| CPU experts (`--ram-experts-mib`) | **Supported, unmeasured** | `cpu_experts()` carries the `situ` path, so CPU experts compute what `ink_situ_rows_kernel` computes |
| Multi-GPU layer split (`--layer-split`) | **Refused** | `inkling2.cpp:79` throws `"--layer-split (the looped core spans the split)"` — a core layer's three visits cannot be divided across cards by physical layer index |
| HIP / ROCm (AMD) | **Unsupported** | `inkling2_kernels.cu:28` hardcodes `cudaStream_t` with no `__HIP__` branch. Sibling `kernels.cu` has one; these kernels do not, so an AMD build fails to compile |
| Metal / macOS (Apple Silicon) | **Unsupported** | there is no Metal variant of the kernels; `inkling2_kernels.cu` is the only implementation |
| CPU-only, no GPU | **Unsupported** | `strata-poly` is only built under `CUDA OR HIP`; attention, short convolution, depth mixing and loop control are card kernels with no host fallback |

Full GPU residency is the configuration most likely to change §8.1: at 8 GB the KV arena and the expert stream compete for
the same VRAM, which is exactly the pressure under which the corruption appears.

---

## 11. Next actions, in priority order

1. **Run the end-to-end cosine gate** (no GPU changes needed to prepare): `STRATA_DUMP_BIN` + `tools/dump_cos.py`
   against `tools/ref_core.py` at `blk.6`, `blk.23` and the final logits, cosine ≥ 0.999. This is the Phase 1 exit
   criterion and the only thing standing between "runs" and "correct".
2. **Find the intermittent corruption.** Instrument `moe_rows()`'s `kStageRows = 32` staging and `copy_seq()`'s per-chunk
   recurrent state; both are exercised only by a long prefill.
3. **Fix the bank leak** behind `5 of 5 at step 42`; one state leaks per request.
4. **Turn on `read_layer_major()`** once per-visit state can move to host memory. This is the largest available prefill win
   and would make §4 obsolete as a baseline.
5. **Close the CUDA-graph invalidation gap** in `shrink_cache()` / `resize_card_cache()` before `L.card` ever becomes 1.
6. **Re-define the speed gate** against something that exists, since the `llama.cpp` comparison is undefined (§6).
7. Decide whether the `STRATA_MMQ_LOG` and `STRATA_GROW_LOG` probes stay. Both are `getenv`-gated and off by default, so
   they are inert in normal use.
8. Then, and only then, the plan's Phase 2 onward: Q8_0 KV, NVFP4 prefill, fusion, CUDA graphs, vision, regression.

---

## 12. Open questions

- Does the corruption survive with **full GPU residency**? Untested, and the most informative single experiment.
- Does it reproduce with `--kv q8_0` at the same length? Q8_0 halves the KV footprint, which changes the pressure.
- Is the `bank is full` leak one state per *request* or one per *prompt chunk*? A chunked prompt would distinguish them.
- Do layers 20–23's Q8_0 experts (no `.scale` tensor) diverge from layers 0–19's NVFP4 at long context? The
  `expert_matrix()` path in `tools/ref_core.py` exists to answer this.
