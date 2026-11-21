# Community benchmark on AMD Radeon RX 7600 XT

Measured on 2026-11-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7600 XT, 16 GiB VRAM
- CPU: AMD Ryzen 7 5700X 8-Core Processor (8 cores, 16 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 10 (10.0.19044); GPU driver: 32.0.32015.7029; CUDA: not read
- PolyStrata: project 0.1.45, commit acd46131; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_m
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly.exe --model Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --max-context 131072 --expert-profile expert-profile.txt --conversation-cache-mib auto --tail-skip 6 --kv-whole
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 74.2 (55.6-82.7) | 0.30 (0.29-0.36) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 702.3 (700.3-706.3) | not measured | 5.73 (5.71-5.76) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 777.4 (777.0-778.1) | not measured | 20.67 (20.65-20.71) |
| agent session, follow-up turns | 15017 (15005-15029) at the last turn | see below | up to 64 | 3 | see below | 67.2 (59.8-76.0) | 7.25 (4.29-9.50) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 255 (243-267) (earlier turns that the cache did not hold; the 14188 (14188-14188) new tokens of the files are read once); first turn (nothing cached) first token: 1.29 s (1.27-1.65).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 370.5 | 1.654 | 55.2 |
| 1 | 4066 | 602 | 3464 | 365.3 | 9.503 | 59.8 |
| 2 | 5701 | 4059 | 1642 | 325.3 | 5.07 | 60.6 |
| 3 | 9122 | 5694 | 3428 | 432.6 | 7.952 | 61.0 |
| 4 | 12081 | 9115 | 2966 | 410.3 | 7.245 | 71.0 |
| 5 | 15017 | 12074 | 2943 | 359.0 | 8.219 | 60.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 245 s.
