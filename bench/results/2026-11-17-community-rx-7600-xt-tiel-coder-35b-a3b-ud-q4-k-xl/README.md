# Community benchmark on AMD Radeon RX 7600 XT

Measured on 2026-11-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7600 XT, 16 GiB VRAM
- CPU: AMD Ryzen 7 5700X 8-Core Processor (8 cores, 16 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 10 (10.0.19044); GPU driver: 32.0.32015.7029; CUDA: not read
- PolyStrata: project 0.1.45, commit acd46131; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: tiel-coder-35b-a3b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly.exe --model Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf --max-context 32768 --expert-profile expert-profile.txt --conversation-cache-mib auto --kv-whole
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
| short chat | 224 (221-229) | 0 | 100 (100-100) | 9 | not measured | 60.2 (42.5-70.4) | 0.55 (0.52-0.66) |
| 4K prompt, fresh | 4025 (4025-4025) | 0 (0-0) | 24 cap | 3 | 762.5 (762.4-768.6) | not measured | 5.30 (5.25-5.30) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 801.5 (800.8-803.1) | not measured | 20.05 (20.02-20.08) |
| agent session, follow-up turns | 15228 (15214-15260) at the last turn | see below | up to 64 | 3 | see below | 52.0 (46.7-63.8) | 6.36 (3.61-8.06) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.8-69.8); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14208 (14208-14208) new tokens of the files are read once); first turn (nothing cached) first token: 1.37 s (1.33-1.55).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 835 | 0 | 835 | 542.4 | 1.549 | 60.2 |
| 1 | 4277 | 844 | 3433 | 481.4 | 7.145 | 50.6 |
| 2 | 5916 | 4319 | 1597 | 380.2 | 4.222 | 47.7 |
| 3 | 9336 | 5964 | 3372 | 419.8 | 8.06 | 48.8 |
| 4 | 12293 | 9371 | 2922 | 439.1 | 6.668 | 52.0 |
| 5 | 15228 | 12339 | 2889 | 437.5 | 6.622 | 46.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 339 s.
