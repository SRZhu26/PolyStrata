# Community benchmark on AMD Radeon + AMD Radeon

Measured on 2026-11-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon + AMD Radeon, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz (4 cores, 8 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.0.0-43-generic; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit a19d955e; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: cyber-tiel-coder-35b-a3b-ud-q4_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly --model Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf --max-context 262144 --expert-profile expert-profile.bench-pool.txt --conversation-cache-mib auto --kv q8_0 --threads 6 --cache-bias 0.5 --vram-reserve-mib 256 --vram-pool
```

Environment of the config: `STRATA_HIP_SPLIT_ATTEND=2`, `STRATA_HIP_PREFILL_FP32=2`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 225 (222-230) | 0 | 100 (100-100) | 9 | not measured | 79.0 (66.9-99.7) | 1.02 (0.95-1.05) |
| 4K prompt, fresh | 4025 (4025-4025) | 0 (0-0) | 24 cap | 3 | 454.7 (453.6-465.6) | not measured | 8.88 (8.66-8.90) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 456.4 (456.2-462.4) | not measured | 35.23 (34.76-35.25) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 417.1 (416.4-420.4) | not measured | 76.90 (76.28-77.03) |
| agent session, follow-up turns | 15226 (15209-15232) at the last turn | see below | up to 64 | 3 | see below | 72.1 (62.8-81.5) | 8.21 (4.41-9.95) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.8-69.8); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14208 (14208-14208) new tokens of the files are read once); first turn (nothing cached) first token: 2.70 s (2.68-2.71).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 836 | 0 | 836 | 313.2 | 2.712 | 74.7 |
| 1 | 4278 | 845 | 3433 | 389.3 | 8.837 | 65.1 |
| 2 | 5932 | 4335 | 1597 | 364.4 | 4.412 | 68.9 |
| 3 | 9347 | 5975 | 3372 | 340.5 | 9.929 | 81.0 |
| 4 | 12305 | 9383 | 2922 | 361.8 | 8.105 | 74.7 |
| 5 | 15232 | 12343 | 2889 | 354.9 | 8.168 | 62.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 530 s.
