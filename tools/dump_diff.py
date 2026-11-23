"""Where do a layer-at-a-time read and a call-at-a-time one first part?

`dbg()` writes a line a dump: `dump pos=... blk=... NAME n=... sum=... abs=... first=... off=...`,
where `off` is the dump's first float in the binary the run wrote beside its log.  A layer-at-a-time
read and a call-at-a-time one take the same tokens in the same calls, so every stage is the same stage:
the first line that differs names the layer and the position where the two reads part.

    python tools/dump_diff.py LAYER_LOG ROW_LOG LAYER.f32 ROW.f32 [--name l_out]

Only the dumps named (all of them by default) are read; each is compared with numpy.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

LINE = re.compile(r"dump pos=(-?\d+) blk=(-?\d+) (\S+)\s+n=(\d+) sum=(\S+) abs=(\S+) first=(\S+) (\S+) (\S+) off=(-?\d+)")


def read_dumps(path, names, min_n=0):
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "dump pos=" not in line:
                continue
            m = LINE.search(line.replace("\x00", ""))
            if not m:
                continue
            pos, blk, name, n, off = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(10))
            if names and name not in names:
                continue
            if n < min_n:
                continue
            out.setdefault((pos, blk, name), (n, off))   # the first pass, not the last
    return out


def read_at(f, off, n):
    f.seek(off * 4)
    return np.frombuffer(f.read(n * 4), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("layer_log")
    ap.add_argument("row_log")
    ap.add_argument("layer_bin")
    ap.add_argument("row_bin")
    ap.add_argument("--name", action="append", help="a stage to compare (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="compare only the first N stages")
    ap.add_argument("--min-n", type=int, default=0, help="skip dumps narrower than this (the warm-up)")
    a = ap.parse_args()
    names = set(a.name) if a.name else set()

    A = read_dumps(a.layer_log, names, a.min_n)
    B = read_dumps(a.row_log, names, a.min_n)
    keys = sorted(set(A) & set(B), key=lambda k: (k[0], k[1], k[2]))
    if a.limit:
        keys = keys[: a.limit]
    print("stages: %d compared (%d layer-major, %d row-major)" % (len(keys), len(A), len(B)))

    by_name, worst, first = {}, (1.0, None), None
    with open(a.layer_bin, "rb") as fa, open(a.row_bin, "rb") as fb:
        for k in keys:
            n, offa = A[k]
            _, offb = B[k]
            x, y = read_at(fa, offa, n), read_at(fb, offb, n)
            nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
            c = float(np.dot(x, y) / (nx * ny)) if nx and ny else 0.0
            by_name.setdefault(k[2], []).append(c)
            if c < 0.999999 and first is None:
                first = (k, c)
            if c < worst[0]:
                worst = (c, k)
    print("\nper stage, worst cosine:")
    for name in sorted(by_name):
        cs = by_name[name]
        print("  %-10s n=%3d  min=%.9f  mean=%.9f" % (name, len(cs), min(cs), sum(cs) / len(cs)))
    if first:
        (pos, blk, name), c = first
        print("\nfirst stage under 0.999999: pos=%d blk=%d %s cosine=%.9f" % (pos, blk, name, c))
    if worst[1]:
        print("worst: %s pos=%d blk=%d cosine=%.9f" % (worst[1][2], worst[1][0], worst[1][1], worst[0]))
    return 0 if worst[0] >= 0.999999 else 1


def read_at(f, off, n):
    return np.frombuffer(read_bytes(f, off, n), dtype=np.float32)


def read_bytes(f, off, n):
    f.seek(off * 4)
    return f.read(n * 4)


if __name__ == "__main__":
    raise SystemExit(main())
