# Five models on a Radeon RX 7600 XT

Measured on 2026-11-17: RX 7600 XT 16 GiB, Ryzen 7 5700X, 63.9 GiB RAM, Windows 10, HIP. Project commit acd46131, project version 0.1.45. These are measurements of the installed configurations, not a controlled comparison of model quality or quantization alone.

BENCH.bat was run sequentially with each of the five strata-*.json configurations and --port 18089. Each benchmark started its own server. Default suite: three repetitions, thinking off, temperature 0, one slot; no --long or --clients. Loading is excluded from suite time.

| Model | Chat decode tok/s | Fresh 16K prompt tok/s | Fresh 16K first token, s | Agent follow-up decode tok/s | Suite time, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Gemma 4 UD-Q4_K_M](../2026-11-17-community-rx-7600-xt-gemma-4-26b-a4b-ud-q4-k-m/) | 66.7 | 939.4 | 17.52 | 58.3 | 203 |
| [Qwen3.6 UD-Q4_K_M](../2026-11-17-community-rx-7600-xt-qwen3-6-35b-a3b-ud-q4-k-m/) | 74.2 | 777.4 | 20.67 | 67.2 | 245 |
| [Qwen3.8 IQ3_S](../2026-11-17-community-rx-7600-xt-qwen3-8-flash-next-iq3-s/) | 22.1 | 187.0 | 85.93 | 15.3 | 1152 |
| [Qwen3.8 Q2_0](../2026-11-17-community-rx-7600-xt-qwen3-8-flash-next-q2-0/) | 11.3 | 110.8 | 144.92 | 10.9 | 2483 |
| [Tiel Coder UD-Q4_K_XL](../2026-11-17-community-rx-7600-xt-tiel-coder-35b-a3b-ud-q4-k-xl/) | 60.2 | 801.5 | 20.05 | 52.0 | 339 |

All speeds are medians. The linked reports contain ranges, per-request results.json and startup log excerpts. All five saved results have empty failures and skipped collections. This does not cover the optional long-context and concurrency tests.

## Limits and metadata discrepancies

- Qwen3.8 uses strata 0.1.59; the other three models use strata-poly 0.1.45. Model sizes, tokenizers, context settings, speculative decoding and cache settings differ. See each report's launch line and saved configuration.
- Reported context is inconsistent with startup logs: IQ3_S reports 32768 while startup says 131072; Q2_0 reports 32768 while startup says 65536; Tiel reports 65536 while startup says 32768. Both original records are retained. These runs exercise only about 16K input tokens and do not validate the larger context limits.
- Q2_0 chat decode ranges from 4.6 to 42.9 tok/s; agent follow-up first-token latency ranges from 25.91 to 231.07 seconds. IQ3_S also varies substantially. The measurements establish the observed variability, not its cause or an inherent ranking of the quantizations.
- GPU identity is recorded as RX 7600 XT in all saved results; the short startup excerpts do not independently establish every hardware detail. Local telemetry code was modified during this installation and is not included in this results-only contribution; the recorded project commit alone does not identify that local change.
- Background workloads, power limits, storage performance and peak memory were not recorded. No answer correctness, coding ability, tool execution or long-context recall checks were run. Agent sessions continue from each model's own answers, so workloads are not token-identical.
- No benchmark was rerun while preparing this submission. Original generated reports and JSON are preserved unchanged.
