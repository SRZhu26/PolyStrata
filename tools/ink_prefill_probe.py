#!/usr/bin/env python3
"""Where an Inkling 2 Mini prefill's time goes: one 8K prompt, one server, one extra engine flag.

The prefill is 1,024-token calls, and the question this answers is whether a call's cost is the experts it moves
into VRAM (which a wider call amortizes over more tokens) or the arithmetic (which it does not).  Run the same prompt
with two `--prompt-batch` values: a per-token cost that falls with the batch is expert movement, one that holds is compute.

    python tools/ink_prefill_probe.py --batch 1024 --batch 512 --out bench/prefill-probe

Each batch gets a fresh server (the cache is loaded the same way every time), the prompt is sent once with
`cache_prompt: false` and four generated tokens, and only the server's own `prompt_ms` is read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import bench_vs_llama as B  # noqa: E402  (its Server, its chat(), its VRAM sampler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=str(ROOT / "bench/prompt-8k-inkling2.txt"))
    ap.add_argument("--cfg", default=str(ROOT / "bench/mma-gate/8k/strata-cfg-Inkling-2-Mini-UD-Q4_K_XL.json"))
    ap.add_argument("--batch", type=int, action="append", required=True, help="a --prompt-batch value, repeatable")
    ap.add_argument("--max-new", type=int, default=4)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt = Path(a.prompt).read_text(encoding="utf-8")
    cfg = json.load(open(a.cfg, encoding="utf-8-sig"))
    machine = {"gguf": cfg["args"][1], "strata_cfg": str(Path(a.cfg).resolve()), "serve": str(ROOT / "serve"),
               "cwd": str(ROOT)}
    if cfg.get("exe"):
        machine["strata_exe"] = cfg["exe"]

    rows = []
    for batch in a.batch:
        name = "pb%d" % batch
        c = {"name": name, "kind": "strata", "args": ["--prompt-batch", str(batch)], "override": {}}
        s = B.Engine(machine, c)
        row = {"prompt_batch": batch, "prompt_n": None, "prompt_ms": None, "prompt_tps": None,
                "pred_n": None, "pred_tps": None, "vram_peak_mib": None, "rss_peak_gib": None}
        try:
            s.start(out)
            row["load_s"] = round(s.load_s, 1)
            r = s.chat(prompt, 0.0, max_tokens=a.max_new)
            row.update({k: r[k] for k in ("prompt_n", "prompt_ms", "prompt_tps", "pred_n", "pred_tps")})
            row["vram_peak_mib"] = s.sampler.vram
            row["rss_peak_gib"] = round(s.sampler.rss / (1 << 30), 2)
            row["status"] = "ok"
        except Exception as e:  # a batch that does not fit is a result, not a crash
            row["status"] = "failed: %s" % e
        finally:
            s.stop()
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out / "runs.json").write_text(json.dumps(rows, indent=1))
        time.sleep(3)


if __name__ == "__main__":
    main()
