# 2x RX 5700 XT (gfx1010) on Linux, a layer split: Cyber-Tiel Coder 35B-A3B UD-Q4_K_XL

Measured 2026-11-15 with the unmodified `tools/strata_bench.py --config <config> --long` (suite 1, three runs a
test, thinking off). It started a fresh server for each configuration. `docs/AMD_HIP.md` says of `strata-poly` on
gfx1010 under Linux that it "has not been tried": this is such a run, on two cards.

## Hardware and software

- **GPUs:** 2x Radeon RX 5700 XT 8 GiB (Navi 10, gfx1010), both used, the layers split over them
  (`"gpu": [0, 1]`, `"layer_split": "auto"`: layers 0 to 21 on card 0, 22 to 39 on card 1). Each on a PCIe 3.0 x8
  link. No Resizable BAR on this board: the kernel says `Detected VRAM RAM=8176M, BAR=256M` for each.
- **CPU:** Core i7-3770 (4 cores / 8 threads; AVX, no AVX2 or FMA), 6 threads for the experts outside VRAM.
- **RAM and storage:** 32 GB DDR3-1600, two channels; the model on a SATA SSD (in the page cache for every run).
- **System:** Ubuntu 26.04.1, Linux 7.0.0-43-generic, ROCm 7.14.0a20260612 from AMD's TheRock wheels
  (`gfx101X-dgpu`), in a Python venv.
- **Displays:** none attached to either card. Nothing else used the cards or more than a few percent of the CPU.
- **Power limits:** the cards' defaults; not read.
- **Engine:** compiled here from source, as `setup.py`'s `build_engine_poly_hip` does it for Linux, with the two
  options this PC needs: `-DCMAKE_HIP_ARCHITECTURES=gfx1010 -DSTRATA_ISA_FLOOR=avx`
  (also `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=OFF`).

| Run | Source | Engine SHA-256 |
| --- | --- | --- |
| [`main-kv-whole`](runs/main-kv-whole/README.md) | `main` at `67dde645`, unmodified | `7beaf234a020edc11110cc309ea9fb3dc4dfcad9238fc195669a344fdac05106` |
| [`pr176-vram-pool`](runs/pr176-vram-pool/README.md) | `67dde645` with PR 175 (a wait for a VRAM mapping on Linux) and PR 176 (`--vram-pool` on a layer split) | `7fabe74c487652f83c746e834f832167316fd5d61d21e54e6dfa3f913074b9d2` |

## Model

- `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf` from
  [peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP),
  revision `b62212407346cb7aafcc8d2788a1844555002fd3`, 22,749,880,640 bytes, SHA-256
  `0bbcf3cc9be4c976bad20e641baf629dad9c178d39ebdc9cd72129179943c06a`. Its draft block is in the file.
- Tokenizer pack: `tools/strata_tokenizer.py --gguf <the file>`.
- Expert profile: each run started from the same copy of a profile the same model's UD-Q3_K_XL file learned on
  another PC (10,495 lines). The engine goes on learning during a run.
- No image encoder in these runs.

## Settings

Both: `--max-context 262144 --kv q8_0 --threads 6 --cache-bias 0.5 --conversation-cache-mib auto`,
`STRATA_HIP_SPLIT_ATTEND=2`, `STRATA_HIP_PREFILL_FP32=2`, one request at a time, the model card's sampling for
fields a client leaves out (the suite sends temperature 0).

- `main-kv-whole`: `--kv-whole --vram-reserve-mib 512`. 4,053 of the 10,496 routed experts in VRAM (7.2 GiB).
- `pr176-vram-pool`: `--vram-pool --vram-reserve-mib 256`. Each card claimed a pool (5,582 and 5,242 MiB); 5,768
  experts in VRAM at load (10.3 GiB).

`--cache-bias 0.5` moves answers a little (docs/BACKENDS.md). Without it, three chats wrote at 56 tokens/s in place
of 72 here, in a separate A/B that is not part of this report.

## Results

Median (lowest to highest; n).

| Run | Chat writes, tokens/s | Chat first token, s | ~4K prompt reads | ~16K prompt reads | ~32K prompt reads | Agent session writes | Agent turn first token, s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `main-kv-whole` | 77.2 (60.5 to 95.2; 9) | 1.24 (1.22 to 1.27) | 430.4 (429.4 to 439.9; 3) | 453.9 (453.5 to 455.2; 3) | 419.4 (419.3 to 420.9; 3) | 75.1 (64.9 to 82.2; 15) | 8.65 (4.96 to 11.60) |
| `pr176-vram-pool` | 79.0 (66.9 to 99.7; 9) | 1.02 (0.95 to 1.05) | 454.7 (453.6 to 465.6; 3) | 456.4 (456.2 to 462.4; 3) | 417.1 (416.4 to 420.4; 3) | 72.1 (62.8 to 81.5; 15) | 8.21 (4.41 to 9.95) |

In both, the agent session reused 69.8% of its follow-up prompt tokens from the conversation cache and read 5
earlier-turn tokens again a session. No request failed. Actual token counts and every request are in each run's
`results.json`.

The two runs are one pass each, not an alternating A/B. Chats here differ by about 3 tokens/s between identical
runs, so only the first-token time and the 4K read separate the two.

## What a reader with these cards should know

- **The default reserve faults on `main` here.** Without `--vram-reserve-mib`, the expert cache grows after the
  first request, and on these cards the engine then stops with `Memory Fault Error ... kernel:
  __amd_rocclr_fillBufferAligned` and `an illegal memory access was encountered` from every request. A named
  reserve (512 above) keeps the cache from growing and avoids it. PR 175 has the cause and a fix.
- **`--vram-pool` on a split needs PR 176**, and on these cards PR 175 under it.
- **The model loads in 2.5 s** from the page cache with `--kv-whole` and in 4.1 s with the pool.
- **One reset, not in these runs:** in another test the same day (fresh prompts up to 116K tokens, `--kv-whole`),
  card 0 took a page fault 5 s into the 116K prompt and the kernel reset it, once in three runs. PR 175 describes it.

## Not measured

Correctness (this is a speed suite), power draw, more than one client at once, pictures, a start with the file read
from the disk, and any other model.
