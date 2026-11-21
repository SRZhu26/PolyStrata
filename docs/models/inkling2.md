# Inkling 2 Mini 26B-A5B (engine scaffolding)

Thinking Machines Lab's [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) in
[Unsloth's `Mainline_Support` GGUFs](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support)
uses the `inkling2_loop` architecture (`ggml-org/llama.cpp` PR `#32694` + `#32704`).

> **Status:** Scaffolding only. `Inkling2::load_geometry()` and KV sizing (FP16 and Q8_0) are wired into
> `src/poly/inkling2.cpp`, `src/poly/engine.cpp`, `CMakeLists.txt` and `setup.py`. Forward kernels are stubbed
> until built and tested on a CUDA/HIP machine. No speeds or outputs have been measured on `strata-poly` yet.
> For the complete tensor inventory, phased kernel implementation/fusion plan, and 4-stage verification plan, see
> **[inkling2_plan.md](inkling2_plan.md)**.

## Architecture and KV cache math

From `Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file` (`ea0c79b2ecd73669e774c8519b79bfabfddbc027`):

- **Looped core**: 24 physical layers (`0..23`), with layers `6..17` run 3× using shared weights and rank-64 loop
  gates → **48 logical layer applications per token** (26.2B total, 3.4B unique active, 5.4B effective active).
- **Attention split**: 20 physical local SWA layers (`sliding_window = 1024`) and 4 global layers (`5, 11, 17, 23`)
  → **40 local SWA + 8 global** logical KV caches per sequence (`n_head = 24`, `n_kv = 8`, `head_dim = 128`).
- **Per-token KV size** (using `strata-poly`'s Q8_0 block layout from [Q8_KV.md](../Q8_KV.md)):
  - **FP16**: `4,096 B/tok/layer` → **32,768 B/tok** (32 KiB/tok) across 8 global layers + **320 MiB** SWA rings
    (`1024 + 1024` prompt-batch rows).
  - **Q8_0**: `2,176 B/tok/layer` (`1,088 B` per K or V row) → **17,408 B/tok** (17 KiB/tok) across 8 global layers
    + **170 MiB** SWA rings (46.88% less KV VRAM).
- **Tensor sizes (from GGUF headers)**:
  - `UD-Q4_K_M` (16.61 GB): 14.78 GB routed experts (`NVFP4`, `Q5_K`, `Q8_0`), 1.83 GB non-expert weights (`Q8_0`,
    `BF16`, `F32`).
  - `UD-Q4_K_XL` (17.63 GB): 15.80 GB routed experts (`NVFP4`, `Q8_0` only), 1.83 GB non-expert weights.

### Estimated VRAM budget on 8 GB VRAM / 32 GB RAM (RTX 5060 Laptop)

With ~7.0 GiB usable VRAM on Windows (~1.71 GiB non-expert weights + ~0.9 GiB scratch/overhead leaves ~4.4 GiB
for KV + hot experts):

| Context | FP16 KV Total | Q8_0 KV Total | Est. VRAM Left for Experts (Q8_0) |
| --- | ---: | ---: | ---: |
| 32K | 1.31 GiB | 0.70 GiB | ~3.7 GiB |
| 64K | 2.31 GiB | 1.23 GiB | ~3.2 GiB |
| **128K (default)** | 4.31 GiB | **2.29 GiB** | **~2.1 GiB** |
| 192K | 6.31 GiB | 3.35 GiB | ~1.0 GiB |
| 256K | 8.31 GiB | 4.42 GiB | ~0 GiB |

`setup.py --family inkling2` defaults to **128K** context, **`--kv q8_0`**, and **`UD-Q4_K_M`** (with
**`UD-Q4_K_XL`** as the second choice).

## Bringup on the RTX 5060 Laptop (Windows, 32 GB RAM, 8 GB VRAM)

### 1. Checkout this branch and download the GGUF

```powershell
git fetch fork
git checkout add-inkling2-mini
```

Install the model files (or pass `--gguf-dir` if you already downloaded them; rename `.gguf_file` to `.gguf`):

```powershell
python setup.py --family inkling2 --model UD-Q4_K_M --context 131072 --kv q8_0 --vision no --yes --no-start
```

### 2. Build `strata-poly` from source (CUDA)

Requires Visual Studio Build Tools (C++), CUDA Toolkit, and CMake:

```powershell
python setup.py --family inkling2 --model UD-Q4_K_M --build --yes --no-start
```

Verify that `Engine::open()` recognizes `inkling2_loop` and prints the 48-layer geometry and Q8_0 KV row size
before hitting the `Inkling2::load_layer()` bringup stub.

### 3. Reference run with `llama.cpp` (for greedy parity)

Until the remaining kernels in `src/poly/inkling2.cpp` are filled in, run Inkling-2-Mini with mainline `llama.cpp`
(including PR `#32704` for per-expert NVFP4 scales on looped passes 2 and 3) at 128K + Q8_0 KV:

```powershell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch origin pull/32704/head:inkling2-loop-fix
git checkout inkling2-loop-fix
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j --target llama-cli llama-server

.\build\bin\Release\llama-server.exe `
  -m ..\models\inkling2-ud-q4_k_m\Inkling-2-Mini-UD-Q4_K_M.gguf `
  --jinja -ngl 99 -c 131072 -ctk q8_0 -ctv q8_0 `
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --port 8001
```

### 4. Remaining `src/poly/inkling2.cpp` bringup checklist

1. `Inkling2::load_layer()`: load non-expert tensors (`attn_q/k/v/gate/output/rel/rel_bias/q_clip/rel_clip/sink_proj`,
   `attn_{k,v,post}_conv`, `attn_k_{conv_proj,inp_norm}` on global layers, `attn_res_query`/`ffn_res_query`,
   `loop_{q,k,v,ffn}_ctrl_{down,forget,gain}`, `loop_msg_query`) and fused `ffn_gate_up_exps.{weight,scale}` +
   `ffn_down_exps.{weight,scale}` (keeping per-expert scales bound on all 3 core passes).
2. `src/poly/kernels.cu`: relative-position attention (`d_rel=16`, `rel_extent=1024`, QK/rel L2 clip, log-scaling,
   value norm without gamma, FP16 + Q8_0 KV readers), causal short conv (`k=4`), rank-64 loop gates, block-6
   attention residuals, and situ softcapped MoE (`4.0`/`25.0`) + final logit softcap (`30.0`).
3. Parity: compare greedy outputs (`temperature=0`) against mainline `llama.cpp` (+ `#32704`) on `UD-Q4_K_M` and
   `UD-Q4_K_XL`, then check `tools/kv_agree.py` for FP16 vs Q8_0 KV.
4. Before any upstream PR: build and verify both CUDA and HIP backends (`AGENTS.md`).
