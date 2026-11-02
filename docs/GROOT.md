# Groot 19B-A3B 261023 (experimental)

[GrootTeam's GGUF](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023-GGUF) uses `qwen35moe`,
already implemented by `strata-poly`. Setup offers its Q4_K_M file and optional BF16 image projector.
This integration has not yet been validated by real inference on Windows, Linux or an RTX 5060.
Matching tensor headers are not an inference correctness or speed measurement.

## Choose it during setup

Run `START-HERE.bat` on Windows or `./setup.sh` on Linux and choose **Groot 19B-A3B 261023** from the model menu.
If another model is already installed, pass `--setup` to return to the setup questions.
Or go straight to this family:

```powershell
python setup.py --family groot
```

Like Tiel Coder and Qwen3.6, setup checks the hardware, offers the shared context menu (8K, 32K, 64K and 128K),
asks whether to download the image encoder, downloads and verifies the selected files, extracts the tokenizer
and writes the run config and launcher. Q4_K_M is currently the only quantization published in this repository.
The default context is 32K, the same as the other `strata-poly` families; an explicit `--context` overrides it.

For an unattended install with a chosen context and images:

```powershell
python setup.py --family groot --model Q4_K_M --context 32768 --vision yes --yes --no-start
python polystrata.py --model groot-q4_k_m
```

For text only, replace `--vision yes` with `--vision no`;
the projector is then neither downloaded nor loaded. An existing model folder can be supplied with
`--gguf-dir "C:\Models\Groot"`. Keep the published filenames in that folder.
Subsequent starts reuse the downloaded files and saved config.

The model download is 11,657,618,560 bytes (11.66 GB decimal); the optional projector adds 902,822,016 bytes
(0.90 GB). Allow extra disk space for the engine, Python packages, tokenizer and installation files.
Both downloads are pinned to revision `2fa1ec958e3bbf7312cf0af97802ff26bfdc5260` and checked by SHA-256.

The engine keeps expert tensors in the model's RAM-backed file cache and caches as many as fit in VRAM.
The GGUF directory contains 10.01 GB of routed experts and 1.63 GB of other weight tensors.
Setup budgets 4 GiB of VRAM beside the experts, including weights, recurrent state, KV rows and scratch.
That budget and its roughly 16 GB RAM estimate beside an 8 GiB card are estimates, not measured peaks.
The budget uses the 32K context, including the draft block's KV rows (22,528 bytes per token over 11 attention layers).
Windows, the display, other applications, image encoding and longer requests need headroom.
The expert cache adapts to available VRAM; total model size is not a full-GPU fit claim.

## Hardware and shared engine features

Groot uses the same `strata-poly` path as Tiel Coder and Qwen3.6, not an RTX 5060-specific engine.
The existing NVIDIA CUDA and AMD HIP build/installation paths, hardware checks, adaptive expert caching,
CPU experts, conversation caching, parallel requests and multi-GPU layer splitting apply.
Backend limitations are unchanged: the image encoder needs the existing CUDA build or CPU placement
where offered, and this does not add a Qwen35 engine to Intel Arc/SYCL.

Use the usual options, for example `--parallel 2`, `--gpus all`, `--vision cpu` and `--context 65536`.
Setup's shared memory estimates decide how much of the experts the GPU can hold.
Larger cards can cache more experts; smaller cards rely more on RAM and CPU work.
No specific card's speed or maximum usable context has been measured for Groot yet.

On Windows 11 with 32 GB RAM and an RTX 5060 8 GB, `--vision yes` automatically places image encoding on the CPU.
Choose a shorter context such as `--context 8192` if needed; `--context 4096` is also available explicitly for
an initial smoke test. These are choices, not a special limit or default for this model.
Image tokens, conversation history and generated output all share the context budget.
Do not treat Windows shared GPU memory or persistent SSD paging as a successful memory fit.

## Chat, tools and images

Setup extracts the tokenizer and embedded chat template from the model itself, not from another Qwen model.
The tokenizer declares `gpt2` with the `qwen35` pretokenizer. The image projector declares
`qwen3vl_merger` and projects to the text model's 2,048-wide embeddings.
Images use the existing server/web upload and OpenAI-compatible image-content path.
CPU image encoding is the automatic choice for an 8 GB card; `--vision gpu` explicitly requests GPU encoding.
Real image-answer correctness and memory use still need validation.

Once running, use the local web app at `http://127.0.0.1:8080`, the OpenAI API under `/v1`, or Anthropic
messages at `/v1/messages`. The API model ID is `groot-19b-a3b-261023-q4_k_m`.
Keep the server bound to `127.0.0.1`; use `--api-key` before exposing it to other machines.

The model card describes thinking by default, API/template-controlled non-thinking mode, and Qwen-style tool calls.
It does not support `/think` or `/nothink` conversation switches. Verify reasoning separation, streamed tool
arguments and tool-result continuation with your client before relying on them.
The card recommends different sampling settings for general thinking, coding and non-thinking requests;
consult it rather than changing PolyStrata's global defaults.

The GGUF contains one MTP draft block. Like Tiel Coder and Qwen3.6, setup leaves detection and speculative decoding
to the shared engine rather than forcing `--no-draft`. To compare ordinary decoding, add `--no-draft` to the
generated model config and restart. The registry reports MTP as disabled while that flag is present.
Groot does not set an approximate tail-skipping override.

## What remains to be measured

- Compare text logits/output, tokenizer IDs and chunked prefill against a compatible reference runtime.
- Exercise both API formats, streaming, thinking/non-thinking, multi-turn tools and model switching.
- Test repeated image/text requests at the selected context lengths, including explicit errors when context is exceeded.
- Record peak dedicated/shared VRAM, process working set/commit, available system RAM, prompt processing and
  generation tokens/s, CPU, build and driver details. Compare ordinary decoding and MTP for correctness and memory use.

No RTX 5060 throughput or real RAM/VRAM peak is published yet. The model's trained context is 262,144 tokens;
its card recommends at least 128K for complex reasoning. A 4K/8K smoke test does not establish that capability
or reproduce the card's benchmark results.
