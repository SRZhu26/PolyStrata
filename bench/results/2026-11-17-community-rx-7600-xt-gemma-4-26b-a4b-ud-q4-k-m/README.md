# Community benchmark on AMD Radeon RX 7600 XT

Measured on 2026-11-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7600 XT, 16 GiB VRAM
- CPU: AMD Ryzen 7 5700X 8-Core Processor (8 cores, 16 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 10 (10.0.19044); GPU driver: 32.0.32015.7029; CUDA: not read
- PolyStrata: project 0.1.45, commit acd46131; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gemma-4-26b-a4b-ud-q4_k_m
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly.exe --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --max-context 32768 --expert-profile expert-profile.txt --draft-model mtp-gemma-4-26B-A4B-it-Q8_0.gguf --conversation-cache-mib auto --kv-whole
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
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 66.7 (55.2-81.0) | 0.28 (0.26-0.34) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 1026.4 (1024.3-1042.1) | not measured | 4.01 (3.96-4.05) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 939.4 (936.3-944.0) | not measured | 17.52 (17.48-17.63) |
| agent session, follow-up turns | 15288 (15267-15289) at the last turn | see below | up to 64 | 3 | see below | 58.3 (53.2-72.1) | 4.60 (2.98-5.64) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.9 % (67.9-67.9); earlier-turn tokens read again per session: 262 (241-263) (earlier turns that the cache did not hold; the 14665 (14665-14665) new tokens of the files are read once); first turn (nothing cached) first token: 0.57 s (0.57-0.69).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 581.2 | 0.69 | 72.9 |
| 1 | 4021 | 389 | 3632 | 882.6 | 4.135 | 55.1 |
| 2 | 5727 | 4014 | 1713 | 556.9 | 3.092 | 56.9 |
| 3 | 9218 | 5720 | 3498 | 706.4 | 4.981 | 53.6 |
| 4 | 12224 | 9211 | 3013 | 658.9 | 4.597 | 58.3 |
| 5 | 15267 | 12217 | 3050 | 543.4 | 5.636 | 55.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 203 s.
