# Ornith 1.5 and Nex-N2.5-mini on one RX 7900 XTX (gfx1100): suite, and PolyStrata vs llama.cpp

Measured on 2026-11-15. Measurement only, no engine changes.

## Hardware and software

- GPU: AMD Radeon RX 7900 XTX (gfx1100), 24 GiB, one card
- CPU / RAM: Ryzen 7 7800X3D (8 cores, 16 threads), 30.6 GiB
- OS: CachyOS Linux, kernel 7.3.4-1-cachyos, ROCm 7.2.4 in `/opt/rocm`, HIP backend
- PolyStrata: main at `67dde645` (`v0.1.45-poly.9-14`), engine compiled on this machine by `./update.sh --backend hip`
- llama.cpp: `llama-server` built from source with HIP, commit `0a66ac7` (version string `0.9.0-dev`), not a release binary
- Background load: nothing else on the GPU. An agent harness that talks to a cloud model ran on the CPU during part of the
  day; its load was not measured. The three `bench_vs_llama.py` configs ran one after another with the GPU otherwise idle.

## What is in this folder

| Folder | Tool | Model file |
| --- | --- | --- |
| `suite-ornith-apex-i-compact/` | `tools/strata_bench.py --runs 3 --long` | Ornith 1.5 35B-A3B APEX-MTP-I-Compact (17.4 GB, the file setup installs for `--family ornith`) |
| `suite-nex-n2-5-mini/` | same | [bartowski/nex-agi_Nex-N2.5-mini-GGUF](https://huggingface.co/bartowski/nex-agi_Nex-N2.5-mini-GGUF) `IQ4_XS` (19.9 GB), installed with `setup.py --custom-gguf` |
| `vs-llama/` | `bench/bench_vs_llama.py`, 3 rounds | the Ornith APEX-MTP-I-Compact GGUF above, read by both engines |

Ornith runs with the install's own settings (`--max-context 65536 --conversation-cache-mib auto --vram-reserve-mib 0 --kv-whole`).
Nex is `qwen35moe` for PolyStrata (`setup.py --inspect` says so); setup matched it to the `qwen36-strata` family with the AMD
defaults (`--kv-whole`). Nex has **no MTP head**: the GGUF has no `mtp.*` tensors, so the engine reads one token at a time.

## Suite results (medians; ranges and per-request data are in the sub-folders)

| Test | Ornith APEX-I-Compact | Nex-N2.5-mini IQ4_XS |
| --- | ---: | ---: |
| short chat, decode tok/s | 153.7 (146.0-186.8) | 113.2 (93.0-114.1) |
| 4K prompt, prompt tok/s | 3,434.7 | 4,106.9 |
| 16K prompt, prompt tok/s | 2,798.7 | 3,065.1 |
| 32K prompt, prompt tok/s / TTFT | 2,188.6 / 14.67 s | 2,261.0 / 14.21 s |
| agent session, decode tok/s | 138.3 (118.2-156.4) | 105.8 (95.0-117.6) |
| agent session, cache hit / re-read per session | 68.6 % / 268 tokens | 68.6 % / 257 tokens |

On this card both models run all experts from VRAM and the 64K context fits (`--kv-whole`). Ornith decodes faster (its MTP block is
used); Nex, without MTP, reads prompts faster (+19.6 % at 4K, +9.5 % at 16K, +3.3 % at 32K).

## PolyStrata vs llama.cpp on the same GGUF (`vs-llama/`)

Run exactly as `docs/COMMUNITY_BENCHMARKS.md` describes: `bench_vs_llama.py run --rounds 3`, then `summary`. Configs: `polystrata`
(the install's config), `llama-mtp` (`--fit on -fa on -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 2`) and `llama`
(the same without speculation), context 36,864 for all three. 256 new tokens per request; the 4K prompt twice and the 32K prompt
once per round, at temperature 0 and 0.7. The rounds alternate the engine order; no errors.

| Config | Prompt | Temp | n | Prompt tok/s | Decode tok/s | TTFT s | Draft acceptance | VRAM peak MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| polystrata | 4K | 0.0 | 6 | 3,515.2 | **136.10** | 1.15 | 0.61 | 18,928 |
| polystrata | 4K | 0.7 | 6 | 3,489.2 | **141.60** | 1.16 | 0.64 | 18,928 |
| llama-mtp | 4K | 0.0 | 6 | 3,222.8 | 117.09 | 1.26 | 0.35 | 18,113 |
| llama-mtp | 4K | 0.7 | 6 | 3,210.7 | 111.34 | 1.26 | 0.30 | 18,116 |
| llama | 4K | 0.0 | 6 | 3,350.5 | 113.26 | 1.21 | - | 16,707 |
| llama | 4K | 0.7 | 6 | 3,345.4 | 113.32 | 1.21 | - | 16,709 |
| polystrata | 32K | 0.0 | 3 | 2,104.5 | 104.70 | 15.23 | 0.66 | 18,938 |
| polystrata | 32K | 0.7 | 3 | 2,074.4 | 101.70 | 15.45 | 0.63 | 18,939 |
| llama-mtp | 32K | 0.0 | 3 | 2,501.0 | 103.59 | 12.81 | 0.37 | 18,116 |
| llama-mtp | 32K | 0.7 | 3 | 2,496.4 | 101.27 | 12.84 | 0.35 | 18,116 |
| llama | 32K | 0.0 | 3 | **2,664.3** | 89.54 | **12.03** | - | 16,709 |
| llama | 32K | 0.7 | 3 | **2,650.7** | 89.39 | **12.09** | - | 16,710 |

VRAM peak is the highest value of the amdgpu counter (`mem_info_vram_used`) seen while the engine ran, including the ~42 MiB the
desktop already used. It is the whole card, not only the engine's share, and it is a polled peak, so short spikes can be missed. At 36,864 context all three configs fit in VRAM.

Computed from the medians above (temperature 0 / 0.7):

| | Decode, PolyStrata vs llama | Decode, PolyStrata vs llama-mtp | Prompt, PolyStrata vs llama | Prompt, PolyStrata vs llama-mtp |
| --- | ---: | ---: | ---: | ---: |
| 4K | +20.2 % / +25.0 % | +16.2 % / +27.2 % | +4.9 % / +4.3 % | +9.1 % / +8.7 % |
| 32K | +16.9 % / +13.8 % | +1.1 % / +0.4 % | -21.0 % / -21.7 % | -15.9 % / -16.9 % |

What this shows on gfx1100, for this file:

- **4K:** PolyStrata decodes 16-27 % faster than both llama.cpp configs. llama.cpp's `draft-mtp` gives little here (+3.4 % / -1.7 %
  against plain llama; draft acceptance 0.30-0.35 against PolyStrata's 0.61-0.64).
- **32K:** PolyStrata and llama.cpp with `draft-mtp` decode about the same (+1.1 % / +0.4 %); both are 13-17 % faster than llama.cpp without
  speculation. The PolyStrata advantage at 4K is gone at this length.
- **32K prompt:** llama.cpp reads the prompt 21-22 % faster than PolyStrata, so the first token comes 3.2-3.4 s earlier (12.0 s vs 15.2-15.5 s).
- **VRAM:** card peak 16.3 GiB with plain llama.cpp, 17.7 GiB with `draft-mtp` and 18.5 GiB with PolyStrata (16,709 / 18,116 / 18,938 MiB); the same at 4K and 32K.

## Limitations

- **API key:** the script needs the two changes in PR #179: it drops the install's `api_key` from the copied config, and it reads VRAM from the amdgpu sysfs counter when `nvidia-smi` is missing. Without it, an install
  with an `api_key` gets HTTP 401 on every `polystrata` request; this run used a checkout with that change.
- **llama.cpp build:** a HIP build from source at `0a66ac7`, not a published release; a release build may differ. llama.cpp's
  `--fit on` chose its own offload and batch settings.
- **Suite:** speed only, thinking off, as the tool does it. The GPU label the tool wrote (`AMD Radeon`) and the OS line were completed
  by hand (card name, ROCm version). Paths in the files are replaced by placeholders.
- **Nex file:** one quantization (IQ4_XS) and one file from one uploader; no comparison with other Nex files.
- **One machine, one day.** Run-to-run ranges are in the sub-folders; clocks and temperatures were not logged.
