# Community benchmark on AMD Radeon RX 7900 XTX (gfx1100)

Measured on 2026-11-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX (gfx1100), 24 GiB VRAM
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 30.6 GiB; storage: not measured
- OS: CachyOS Linux, kernel 7.3.4-1-cachyos (amdgpu in-kernel driver); ROCm 7.2.4 in /opt/rocm; HIP backend
- PolyStrata: project 0.1.45, commit 4eab97d5; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: nex-agi_nex-n2.5-mini-iq4_xs
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly --model nex-agi_Nex-N2.5-mini-IQ4_XS.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib auto --tail-skip 6 --prompt-batch 2048 --kv-whole
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 44 (41-49) | 0 | 100 (100-100) | 9 | not measured | 113.2 (93.0-114.1) | 0.12 (0.11-0.18) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 4106.9 (3999.0-4123.3) | not measured | 1.00 (1.00-1.03) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3065.1 (2724.0-3067.5) | not measured | 5.26 (5.26-5.92) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2261.0 (2245.2-2340.5) | not measured | 14.21 (13.72-14.30) |
| agent session, follow-up turns | 15062 (15041-15084) at the last turn | see below | up to 64 | 3 | see below | 105.8 (95.0-117.6) | 1.23 (0.60-1.39) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 257 (236-278) (earlier turns that the cache did not hold; the 14203 (14203-14204) new tokens of the files are read once); first turn (nothing cached) first token: 0.25 s (0.25-0.35).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 607 | 0 | 607 | 1852.3 | 0.348 | 117.3 |
| 1 | 4063 | 606 | 3457 | 3469.8 | 1.021 | 110.1 |
| 2 | 5716 | 4062 | 1654 | 2753.9 | 0.626 | 110.5 |
| 3 | 9151 | 5715 | 3436 | 2776.3 | 1.263 | 105.8 |
| 4 | 12134 | 9150 | 2984 | 2476.6 | 1.232 | 102.4 |
| 5 | 15084 | 12133 | 2951 | 2174.0 | 1.383 | 96.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 101 s.
