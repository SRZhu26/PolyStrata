#!/usr/bin/env python3
"""Where the continuation stops being English: one server, the same text cut shorter and shorter.

The 16k run's answer cycles one sentence, while at short context the model answers correctly, so the
defect is a long-context one.  This starts the server once and sends prefixes of one prompt at several
lengths, greedy (the most sensitive setting), so the length where it breaks is bracketed.  What that length
is says which mechanism: the sliding window's ring, the relative bias's extent, or the log-scaled positions
whose ramp starts at `log_scale_floor` (8192).

    python tools/probe_context.py --model PATH.gguf --prompt FILE --lengths 512,1024,2048,4096,8192,12288

The prompt is cut by characters, which is close enough for bracketing (about 4.5 chars a token here) and
is exact in what the model reads: the text is the same text, only shorter.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_128k_local import CUDA_BIN, ROOT, chat, start, stop, strata_config
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192,12288")
    ap.add_argument("--tokenizer", default=str(ROOT / "bench" / "tok-inkling2" / "tokenizer"))
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--kv", default="fp16")
    ap.add_argument("--extra", default="", help="extra strata args, space separated")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "bench" / "probe-context"))
    a = ap.parse_args()

    text = Path(a.prompt).read_text(encoding="utf-8")
    lengths = [int(x) for x in a.lengths.split(",")]
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfg = strata_config(a.model, out, a.tokenizer, a.context, a.kv, a.threads,
                        a.extra.split() if a.extra else [])
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
                PATH=str(CUDA_BIN) + os.pathsep + os.environ.get("PATH", ""))
    proc, log, load_s, logname = start([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                                        "--config", str(cfg), "--port", "8191"], out, "probe", env)
    if not proc:
        print("DID NOT START (%s)" % logname, flush=True)
        return 1
    try:
        chat("Say hello.", 4)                       # warm-up, discarded
        for n in lengths:
            cut = text[:int(n * len(text) / 16343)] if n < 16343 else text
            r = chat(cut, a.gen)
            print("\n=== %d chars (~%d tokens): prompt_n=%s ===" % (len(cut), n, r.get("prompt_n")), flush=True)
            print((r.get("text") or "").strip()[:400], flush=True)
    finally:
        stop(proc, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
