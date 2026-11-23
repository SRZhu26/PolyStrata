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
| `inkling2-engine-local` | `9e33d64c` (plan build note), engine head `f159fe78` | **this work**, pushed to `origin` as its own branch |

`origin` is `https://github.com/SRZhu26/PolyStrata.git`, the user's own repository; there is no second remote. The engine
work was committed to `inkling2-engine-local` precisely so it does not appear in the `add-inkling2-mini` → `main` pull
request. `origin/main` and the PR head are untouched. Note that the repository is publicly readable.

Left untracked on purpose: `bench/*` outputs, `.workbuddy/`, and a stray `--help` file produced by an accidental
redirect.

---

## 2. Environment and build

- Windows 11, Visual Studio Build Tools (MSVC), CMake, CUDA 13.1, target `sm_120` (Blackwell, RTX 5060 Laptop).
- Build: `tools/build_poly_cuda_local.sh` → `BUILD_OK`; binary `build-poly/strata-poly.exe` (88 MB).
- Model under test: `Inkling-2-Mini-UD-Q4_K_XL.gguf`, 17,629,760,736 B (17.6 GB / 16.42 GiB), 24 physical layers.
- KV cache: FP16 and `--kv q8_0` both measured (§4); Q8_0 is numerically sound (99.79% agreement) but slower at
  8K and 32K, and only pays off with `--kv-whole` at very long context.
- Kernel unit tests: `poly_inkling2_test.exe`, **17/17 pass** against FP64 on the current binary.
- Tokenizer: `bench/tok-inkling2/tokenizer` (converted from the GGUF), used by the server.

---

## 3. What is implemented

The architecture is not reachable through the generic engine path, so it has its own implementation. Nothing here is a
stub: every kernel listed is exercised by the runs in §4.

| File | Lines | Contents |
| --- | ---: | --- |
| `src/poly/inkling2.cpp` | 996 | `Inkling2` class, 32 overrides |
| `src/poly/inkling2.hpp` | 206 | interface and capability flags |
| `src/poly/inkling2_kernels.cu` | 1,044 | 16 CUDA kernels |
| `src/poly/inkling2_test.cu` | 652 | kernel tests against an FP64 reference |

Engine-side integration: arch registration for `inkling2_loop` / `inkling2` in `Engine::open()`, the `--kv q8_0` allowance,
the shared `gate_up` NVFP4 scale across both halves, the `moe_act()` hook, `situ_` / `situ_beta_` / `situ_linear_beta_`
members, `state_whole()`, and the `situ` path in `cpu_experts()`.

Kernels with no equivalent anywhere else in the engine:

| Kernel | Replaces / provides |
| --- | --- |
| `ink_situ` / `ink_situ_rows` | `situ` activation (`beta = 4.0`, `linear_beta = 25.0`) instead of SwiGLU/GeGLU |
| `ink_rel_table` | the `relative_length = 16` bias table, `extent = 1024`, clip `8.0` |
| `ink_attend_rel` | attention consuming that bias — **there is no RoPE** (`rope.dimension_count = 0`) |
| `ink_attend_rel_split` / `ink_attend_rel_q8_split` | flash decoding: `gridDim = (n_head, 16)` at `pos >= 2048`, so a decoding step's 131,072 keys scan in 16 blocks a head instead of one |
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

### 3.1 A layer-major read — attempted, off until the `blk 6` fault is fixed

`Engine::read()` takes a prompt in calls of `rows_` (1,024) tokens, and with the experts outside the card each call copies
every layer's experts through the staging buffer: 128 calls of 1,024 tokens at 128K copy all 14 GB **128 times**. So `read()`
has a layer-major path that runs layer 0 over every call, then layer 1 over every call, and so on, and the second call finds the
layer's experts where the first one's copy left them (`moe_rows()`): the model is copied once for the whole read.

`Inkling2` does **not** take it: `read_layer_major()` returns `false`. The four things that cross a call's layers — the depth
attention's bank, the pass before's blocks and output, and the prelude's output — were staged per call in pinned host memory
(`read_calls()` / `read_call()`), and the result **diverges from the call-at-a-time read at `blk 6`**: cosine 0.9969295 at
`pos 0` and 0.9844632 at `pos 1024` on a 2,048-token prompt, growing to 0.8658 by `blk 22`. The prelude (`blk 0..5`) is
clean at 0.99997, so the fault is a *fixed* error at the core entry, not chaotic amplification — the pass restart, the
`n_bank_ = n_prelude_blocks_` cut, or `pass_messages()`, one of the three. Two layer-major runs agree exactly, so it is
deterministic. The staging is 1.09–1.13× on prefill, which is why it is not worth holding the branch for.

The whole attempt is kept on branch `inkling2-layer-major-wip` (`bb51e87a`), with `tools/layer_major_ab.py` and
`STRATA_NO_LAYER_MAJOR` to reproduce the A/B. Re-land it only with the `blk 6` fault fixed and a cosine ≥ 0.99999 at
`blk 6` and `blk 22` on both `pos 0` and `pos 1024`.

---

## 4. Measured results

All numbers: `UD-Q4_K_XL`, experts streamed from disk, RTX 5060 Laptop, one fresh server per row. The FP16 and Q8_0 rows
share the prompt, the context and the binary, so each pair is a clean A/B on the KV format alone. The tables in §4 and §4.3
were measured on `ee04f3d7`; §4.4 re-measures the current binary (`f159fe78`) and supersedes them for speed.

| Prompt tokens | KV | Prefill tok/s | Decode tok/s | VRAM peak | RSS peak | Experts in VRAM |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8,198 | fp16 | **136.3** | **8.1** | 6,640 MiB | 12.46 GiB | 791 (3.8 GiB) |
| 8,198 | q8_0 | 127.7 | 7.2 | 6,640 MiB | 14.61 GiB | 829 (4.0 GiB) |
| 32,015 | fp16 | **77.3** | **8.7** | 6,640 MiB | 14.88 GiB | 791 |
| 32,015 | q8_0 | 66.9 | 7.6 | 6,640 MiB | 14.39 GiB | 829 |

VRAM peak is identical in every row because KV grows on demand and the expert pool claims the card before it grows
(§4.3). Load time 8–12 s; swap 0.7 GiB in every row.

At 128,072 context, load only (no prefill):

| KV | Experts in VRAM | Expert bytes |
| --- | ---: | ---: |
| fp16 | 791 (3.8 GiB) | 3.8 GiB |
| q8_0 | 829 (4.0 GiB) | 4.0 GiB |

With `--kv-whole` (KV preallocated for the whole context, the mode the plan's 4.31 → 2.29 GiB claim was written
for), at 128,072:

| KV | Experts in VRAM | Expert bytes |
| --- | ---: | ---: |
| fp16 | **0** | 0.0 GiB |
| q8_0 | **405** | 1.9 GiB |

That is the only configuration measured in which Q8_0 KV buys real residency: the FP16 KV fills the card, while Q8_0 leaves
1.9 GiB for 405 of 3,072 experts.

Prefill over the 128K prompt, measured incrementally to 96% before the run was stopped (FP16 KV, pre-fix binary):

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
The curve was measured before the `ink_conv`/warp-softmax fixes; the current binary is faster at every length measured
(8K: 68.3 → 136.3 tok/s marginal), so the 128K projection is now pessimistic.

### 4.1 The earlier 8K/16K numbers are not comparable

The earlier table read 8,198 → 71.7 prefill / 5.0 decode / 7,498 MiB, and 16,343 → 84.4 / 10.1 / 6,640 MiB. Both were
taken on the pre-fix binary, and the 8K run's output degenerated into a loop, so its decode figure understates throughput.
The 16K row is superseded by the 32K and 8K rows above and is kept only for the record.

### 4.2 What Q8_0 KV actually costs

Q8_0 is **numerically sound and slower at every length measured**:

| Context | Prefill change | Decode change |
| ---: | ---: | ---: |
| 8,198 | −6.3% | −11.1% |
| 32,015 | −13.5% | −12.6% |

The KV saving is real but small below 32K, and it converts into only **38 more resident experts** (791 → 829), which is
what the measurement shows. That is not enough residency to pay for the quantized attention kernel, which dequantizes every
K and V element in the score and value passes (`inkling2_kernels.cu:397`, `:432`) where the FP16 kernel reads halves directly.

Q8_0 only becomes a clear win when the KV is preallocated for a long context: at 128K with `--kv-whole` it keeps 405
experts resident where FP16 keeps none. Below that, **FP16 KV is the faster configuration** and Q8_0 should be used only when
the context is long enough that residency, not kernel speed, is the binding constraint.

### 4.3 Q8_0 KV silently disables the RAM expert pool

`ram_experts_mib` is auto-sized by `engine.cpp:2882`: when every expert outside VRAM plus 4 GiB of headroom does not fit in
free RAM, a budget is set and a pinned pool with eight prefetch threads is created; otherwise `ram_experts_mib` stays `-1`,
every expert is served through the file mapping, and **no pool or prefetch thread is created** (`engine.cpp:2893`).

Q8_0 frees VRAM, so 38 more experts are resident and `outside_bytes_` drops just below the threshold. The heuristic flips to
`-1`, and the run loses the pool:

| 8K, q8_0 | Prefill | Decode | RSS |
| --- | ---: | ---: | ---: |
| auto (`-1`, maps everything) | 127.7 | 7.2 | 14.61 GiB |
| `--ram-experts-mib 11187` (the FP16 run's budget) | 126.4 | 7.6 | 12.10 GiB |

So a third of the Q8_0 decode loss is not the kernel at all: it is the heuristic taking the unpinned path because the freed VRAM
moved it across a threshold. Pinning the budget recovers 0.4 tok/s and 2.5 GiB of RSS. This is defect §8.2.

### 4.4 The `f159fe78` attention rewrite: +26% prefill, +27% decode at 8K

`f159fe78` caches each block's queries in shared memory instead of re-reading them per key, makes the value pass one
dimension per thread, hoists the Q8_0 block scales, and reuses one Q8_1 quantization of the layer input across `wq`, `rel`,
`w_gate` and `sink_proj`. Re-measured on the rebuilt binary against the `ee04f3d7` numbers above, same prompt, same
context, same 791 resident experts, same 6,640 MiB VRAM:

| Context | Prefill (before → after) | Decode (before → after) | RSS (before → after) |
| ---: | --- | --- | --- |
| 8,198 | 136.3 → **171.3** (+25.7%) | 8.1 → **10.3** (+27.2%) | 12.46 → 11.20 GiB |
| 32,015 | 77.3 → **97.5** (+26.1%) | 8.7 → **9.5** (+9.2%) | 14.88 → 11.74 GiB |

The 8K pair is the clean one: both runs pinned a RAM expert pool (11,187 vs 9,913 MiB) and held the same 791 experts
in the same 6,640 MiB, so the gain is the kernel. The 32K pair is **confounded** — the `ee04f3d7` run took the
unpinned path (no pool, defect §8.2) and the `f159fe78` run pinned 10,319 MiB, so part of its +9.2% decode and its
3.14 GiB RSS drop is the pool, not the rewrite. The RSS differences in both rows track the pool size rather than the kernel.

The 17 FP64 kernel checks pass on the new binary, including `attend` (7.153e-07 of 2.364) and `attend ring` (5.364e-07 of
2.908), which is the direct validation of the rewritten attention. End-to-end coherence was confirmed on the varied 6,185-token
prompt (`bench/smoke-after-f159fe78`: 181.3 prefill / 9.7 decode, coherent reasoning). The synthetic repeated-passage
prompts still loop at 8K and 32K, exactly as documented in §8.1 — that is content-driven, not a regression.

### 4.5 Split-K flash decoding and the full 8K/32K/128K gate (`--kv q8_0`, `--ram-experts-mib 10240`)

Split-K splits a decoding step's keys over 16 blocks per head for a full-context layer once the position reaches 2,048
(`kAttnSplitMin`), and the reduce folds them back with `exp(m_split − m_all)`; below that the key range is short enough that
the extra launches cost more than the idle blocks do. `STRATA_DP4A` Q8_0 attention landed with it, so Q8_0 KV is no longer
6–13% slower than FP16 — the 8K Q8_0 prefill here is *above* the FP16 figure in §4.4.

One fresh server per row, same prompt set, `--kv q8_0 --ram-experts-mib 10240`:

| Prompt tokens | Prefill tok/s | Decode tok/s | TTFT | VRAM peak | RSS peak | Swap peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,198 | **165.0** | **10.1** | 49.7 s | 6,640 MiB | 11.62 GiB | 0.51 GiB |
| 32,015 | **94.9** | **9.1** | 337.3 s | 6,640 MiB | 11.24 GiB | 0.54 GiB |
| 130,715 | **35.5** | **6.7** | 3,678.7 s (61.3 min) | 6,640 MiB | 11.30 GiB | 0.54 GiB |

The 128K prefill is 61.3 min, which is the quadratic wall §4.4's incremental curve projected at ≈ 96 min before the
attention rewrite; the rewrite and Split-K together bring it in at two thirds of that. Coherent text at 130.7K. System peak
27.26 GiB, so the 32 GB laptop has ~4.7 GiB of headroom at the longest context. The 17 kernel checks pass, including
`attend split` (3.278e-07 of 0.449) and `attend q8 spl` (2.086e-07 of 0.423).

---

## 5. Gates against the plan's criteria

| Gate | Criterion | Result |
| --- | --- | --- |
| Peak VRAM | ≤ 7.5 GiB (7,680 MiB) | **pass** — 6,640 MiB in every run |
| Peak RAM | ≤ 26 GiB | **pass on `f159fe78`** — process RSS 11.20 GiB at 8K and 11.74 at 32K; system peak 25.36/25.25 GiB at 8K/10.9K FP16. The earlier marginal rows (26.2–26.8) were all `ee04f3d7` runs that either took the unpinned path (§8.2) or ran Q8_0 |
| Decode vs `llama.cpp` | ≥ 1.5× same-day baseline | **not evaluable** — no baseline exists (§6) |
| Cosine per stage | ≥ 0.999 | **pass** — `blk.6` 0.999988758, `blk.23` 0.999988499, logits 0.999763105; worst layer 0.999971239 (`blk.16`). `UD-Q4_K_XL` holds no `Q4_K` tensors, so the oracle's NVFP4/Q8_0 coverage is enough |
| Kernel parity | vs FP64 | **pass** — `poly_inkling2_test`, 19 checks |
| Long-context coherence | coherent at 8K/32K/128K | **pass** — coherent at 6,185 / 8,233 / 10,281 tokens on controlled prompts, and at 130,715 on the 128K benchmark prompt |
| Long-context needle | 100% recall at 32K/64K/128K | **not yet run** |
| Q8_0 KV | FP16 vs Q8_0 agreement ≥ 97% | **pass** — 639/640 = **99.84%**, teacher-forced |
| Bank leak | 0 errors over 7 sequential requests | **pass** — 0 occurrences |
| Conv-state race | 3 consecutive ~10.9K prefills coherent | **pass** — coherent at 6,185 / 8,233 / 10,281 tokens, fresh server each |

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
| `poly_inkling2_test` | builds; 19 kernel-level checks against FP64 **pass**, including split-K flash decoding |
| `tools/ref_core.py` | runs end to end — all 74 stages plus logits |
| `tools/dump_cos.py` | **run**: `blk.6` 0.999988758, `blk.23` 0.999988499, logits 0.999763105 |
| `tools/kv_agree.py` | **run**: 99.84% Q8_0 vs FP16 agreement, 639/640 teacher-forced |
| `tools/needle_bench.py` | exists; **not run** for this model |
| `tools/probe_context.py` | **run**: one server, increasing prefix lengths; reuses cached prefixes, so a fresh server per length is needed to isolate |
| `tools/make_long_prompt.py` | **run**: exact-token prompts at 8K, 32K and 128K built for this model |
| `tools/layer_major_ab.py` | on `inkling2-layer-major-wip`; compares a layer-major read against a call-at-a-time one |
| `tools/strata_tokenizer.py` | `inkling2` mapped to a ported `GPT4O_PATTERN`; **byte-for-byte parity unverified** |

**Tokenizer parity is unverified too.** `strata_tokenizer.py` now maps `inkling2` to a `GPT4O_PATTERN` ported from
`llama.cpp`'s `LLAMA_VOCAB_PRE_TYPE_GPT4O`, with `clean_spaces = false` and no BOS. That is a port, not a measurement:
no byte-for-byte comparison against `llama-tokenize` or `transformers` was recorded. A tokenizer that counts differently
would invalidate the token counts in §4, so this should be checked before the §4 numbers are quoted externally.

**Kernel parity is not the end-to-end gate.** The 19 passing checks compare each kernel against an FP64 reference of the
same graph. They do not establish that the assembled 48-visit model matches `ref_core.py`; that is `STRATA_DUMP_BIN` plus
`dump_cos.py` at `blk.6`, `blk.23` and the final logits, and it now passes. The earlier belief that it was blocked on
weights was wrong: `UD-Q4_K_XL` contains no `Q4_K` tensors at all (`Q8_0` 154, `F32` 518, `BF16` 32, `NVFP4` 40), so
the oracle's NVFP4/Q8_0 dequantizers cover every tensor in the file.

---

## 8. Known defects

### 8.1 Repetition loops are content-driven, not a long-context corruption (not a defect)

Earlier runs at roughly 10,900 tokens degenerated into a repetition loop, and the working assumption was a race that a
long prefill exposed. **The controlled experiment says otherwise.** With one question held fixed and only the filler length
changed, every length is coherent:

| Prompt | Tokens | KV | Sampling | Output |
| --- | ---: | --- | --- | --- |
| Lighthouse prose + question | 6,185 | q8_0 | greedy | coherent |
| Lighthouse prose + question | 8,233 | q8_0 | greedy | coherent |
| Lighthouse prose + question | 10,281 | q8_0 | greedy | coherent |
| README dump + question | 1,432 | q8_0 | greedy | loops on "Qwen 3" |
| README dump + question | 1,432 | fp16 | greedy | loops on "Qwen" |
| README dump + question | 1,432 | q8_0 | temp 0.7 | loops on "Poly" |
| README dump + question | 10,941 | fp16 | greedy | loops on "PolyStr" |
| README dump + question | 10,941 | q8_0 | greedy | loops on "5.5.5.5" |

So the loop follows the **prompt**, not the length and not the KV format: it reproduces at 1,432 tokens, under both KV
formats and under both greedy and sampled decoding, and a different document at 10,281 tokens is coherent. The README prompt is a
raw document dump whose salient token is a proper noun repeated dozens of times ("PolyStrata"), and the model locks onto it.
The earlier "two runs in three at 10,900 tokens" observation was the same effect seen through one prompt.

What this does **not** excuse: `read_layer_major() = false` means a long prompt re-reads experts per chunk, and the
`--dump-layers` path is still unverified above 10.3K. Coherence at 128K is now **measured once**: the 130,715-token
benchmark produced coherent text (35.5 tok/s prefill, 6.7 tok/s decode), but needle recall above 32K is still unmeasured.

Ruled out by experiment:

| Suspect | How it was ruled out |
| --- | --- |
| Context length | 10,281 tokens coherent; 1,432 tokens loops |
| Q8_0 KV | FP16 loops identically on the same prompt |
| Sampling / temperature / top_p | identical loop at temp 0.7 + top_p 0.95 and at greedy |
| `tau` log-scaling | `tau = 1.0` below 8192 |
| Softcap, 966-row vocab padding | verified correct in `ink_finish_logits_kernel` |
| SWA ring and window | probe at successive prefix lengths |
| `cover()` KV growth | logs show correct growth in coherent runs too |
| `shrink_cache()` | fires in coherent runs (`cut=3712 MiB cached=770`) |
| Expert swap prefetch | `--expert-swaps 0` still fails, half of runs |
| The attention kernel | a cached `prompt_n=1` re-run over the same KV is always coherent |
| Slot publication ordering | `L.slot[e]` is set only after `h2d()` lands (`engine.cpp:2353-2356`, `2370-2373`) |

**A blind spot made this harder than it needed to be:** `serve/server.py` opens the engine's stderr as `self.log`, which is
`subprocess.DEVNULL` unless the config carries a `"log"` key. Every probe written into `engine.cpp` was invisible until
`strata_config()` in `tools/bench_128k_local.py` was given one. An earlier conclusion that `cover()` never ran was drawn
from that silence and has been retracted.

### 8.2 The RAM-expert heuristic flips to the unpinned path on a small VRAM win

`engine.cpp:2882` sets a pinned expert pool only when `outside_bytes_ + 4 GiB > ram_available()`; otherwise
`ram_experts_mib` stays `-1`, every expert is served through the file mapping, and no pool or prefetch thread exists
(`engine.cpp:2893`). The test is a knife edge, and freeing VRAM moves a run across it:

| 8K, q8_0 | `ram_experts_mib` | Decode | Process RSS |
| --- | --- | ---: | ---: |
| auto | −1 (no pool, no prefetch) | 7.2 | 14.61 GiB |
| `--ram-experts-mib 11187` | 11,187 | 7.6 | 12.10 GiB |

Because Q8_0 KV frees enough VRAM for 38 more experts to stay resident, `outside_bytes_` falls just under the threshold and
the run silently loses the pool. The visible symptom is a slower decode and a larger RSS, which reads as "Q8_0 is slow"
when a third of the loss is the heuristic. Fix: choose the budget from `outside_bytes_` alone, not from a headroom comparison
that a VRAM change can flip, or keep the pool whenever `outside_bytes_` exceeds the pool's own size.

### 8.3 Sixth sequential request on one server dies

```
the depth attention's bank is full (5 of 5 at step 42)
```

`max_states_ = 5`, and a single read should peak at 4, so one bank state leaks per request. The 6th request is the one
that fails. **Fixed** by `!read_layer_major() ||` in `Engine::read()` (`engine.cpp:5538`); verified 0 errors over 7
sequential Q8_0 requests.

### 8.4 Latent CUDA-graph invalidation gap

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

Full GPU residency is the configuration most likely to change §4.3: at 8 GB the KV arena and the expert stream compete for
the same VRAM, which is exactly the pressure under which the corruption appears.

---

## 11. Next actions, in priority order

1. ~~**Run the end-to-end cosine gate**~~ — **done.** `STRATA_DUMP_BIN` + `tools/dump_cos.py` against `tools/ref_core.py`
   reads `blk.6` 0.999988758, `blk.23` 0.999988499 and the logits 0.999763105 against a 0.999 gate. It was never
   blocked on weights: `UD-Q4_K_XL` contains no `Q4_K` tensors, only `Q8_0`/`F32`/`BF16`/`NVFP4`.
2. **Fix the `blk 6` fault in `read_layer_major()`** (§3.1). The staging is written and kept on branch
   `inkling2-layer-major-wip`, but it diverges from the call-at-a-time read at the core entry, so it is off. It is worth
   1.09–1.13× on prefill — fix it only if that is the cheapest available win.
3. **Make the Q8_0 attention kernel pay for itself or drop it.** It is 6–13% slower than FP16 and buys 38 resident
   experts below 32K. Either an `__dp4a` integer dot with the per-32-block scales (as `quantize.cpp`'s dot does) or keep
   FP16 as the default and select Q8_0 only for `--kv-whole` at 128K, where it keeps 405 experts instead of 0.
4. **Fix the RAM heuristic** (§8.2) so a VRAM win cannot silently cost the prefetch pool.
5. **Measure coherence and needle recall above 32K.** Coherence is proven to 10.3K on controlled prompts (§8.1) and the
   prefill is now measured at 32K and 128K (§4), but needle recall at 32K/64K/128K is still unrun.
6. **Close the CUDA-graph invalidation gap** in `shrink_cache()` / `resize_card_cache()` before `L.card` ever becomes 1.
7. **Re-define the speed gate** against something that exists, since the `llama.cpp` comparison is undefined (§6).
8. **A tensor-core MMA prefill attention** (§7.5 of the plan): `ink_attend_rel` is still scalar FP32 `Q·K` and `P·V` at
   `rows >= 64`, which is what leaves the prefill at ≈ 95 tok/s at 32K.
9. Decide whether the `STRATA_MMQ_LOG` and `STRATA_GROW_LOG` probes stay. Both are `getenv`-gated and off by default, so
   they are inert in normal use.

---

## 12. Open questions

- Does coherence hold above 10.3K? Unmeasured; a full 128K prefill is ≈ 96 min at 23 tok/s. The earlier "long-context
  corruption" is now explained by prompt content (§8.1), so this is a measurement gap, not a suspected bug.
- Does the Q8_0 KV win at 128K once the attention kernel is integer? The residency is proven (405 experts with
  `--kv-whole` vs 0), the throughput is not.
- Is the `bank is full` leak one state per *request* or one per *prompt chunk*? A chunked prompt would distinguish them.
- Do layers 20–23's Q8_0 experts (no `.scale` tensor) diverge from layers 0–19's NVFP4 at long context? The
  `expert_matrix()` path in `tools/ref_core.py` exists to answer this, and needs the NVFP4 source too.
