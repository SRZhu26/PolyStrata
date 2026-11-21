# Community benchmark on AMD Radeon RX 7900 XTX (gfx1100)

Measured on 2026-11-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX (gfx1100), 24 GiB VRAM
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 30.6 GiB; storage: not measured
- OS: CachyOS Linux, kernel 7.3.4-1-cachyos (amdgpu in-kernel driver); ROCm 7.2.4 in /opt/rocm; HIP backend
- PolyStrata: project 0.1.45, commit 67dde645; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: ornith-1.5-35b-a3b-apex-i-compact
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly --model Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib auto --vram-reserve-mib 0 --kv-whole
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 153.7 (146.0-186.8) | 0.15 (0.14-0.16) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3434.7 (3290.2-3625.0) | not measured | 1.20 (1.14-1.25) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2798.7 (2789.6-2901.0) | not measured | 5.77 (5.56-5.79) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2188.6 (2180.2-2204.4) | not measured | 14.67 (14.56-14.73) |
| agent session, follow-up turns | 15030 (15016-15055) at the last turn | see below | up to 64 | 3 | see below | 138.3 (118.2-156.4) | 1.34 (0.67-1.47) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 268 (254-293) (earlier turns that the cache did not hold; the 14188 (14188-14188) new tokens of the files are read once); first turn (nothing cached) first token: 0.26 s (0.25-0.33).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1958.2 | 0.334 | 174.3 |
| 1 | 4056 | 602 | 3454 | 3113.7 | 1.134 | 148.1 |
| 2 | 5693 | 4049 | 1644 | 2472.6 | 0.696 | 138.7 |
| 3 | 9111 | 5686 | 3425 | 2606.1 | 1.342 | 137.5 |
| 4 | 12084 | 9104 | 2980 | 2244.0 | 1.357 | 130.1 |
| 5 | 15016 | 12077 | 2939 | 2035.2 | 1.469 | 118.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 100 s.
