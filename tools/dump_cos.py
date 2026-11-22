#!/usr/bin/env python3
"""The cosine similarity of the engine's dumps to an FP64 reference's, stage by stage.

The engine writes a line per dump under `--dump-layers`, and with `STRATA_DUMP_BIN` set it appends that
dump's floats to a file and puts where they begin on the line (`off=`).  `tools/ref_prelude.py` writes the
same for its own vectors under `STRATA_REF_BIN`, so a stage both name has can be compared in full, not just by
its sum, its absolute sum and its first three values.

    STRATA_DUMP_BIN=eng.f32 build-poly/strata-poly.exe ... --dump-layers 2> eng.txt
    STRATA_REF_BIN=ref.f32 python tools/ref_prelude.py FILE.gguf TOKEN > ref.txt
    python tools/dump_cos.py eng.txt eng.f32 ref.txt ref.f32

A stage is matched on its name and its layer: the engine's `l_out` at blk.5 is the reference's `l_out` at blk.5.
A window layer's l_out, a full-context layer's and the token embedding's are the stages a reference for a
single token at position 0 can give.  The plan's gate is a cosine of at least 0.999 for each.
"""
import re
import sys

import numpy as np

LINE = re.compile(r"dump pos=(-?\d+) blk=(-?\d+) (\S+)\s+n=(\d+) sum=\S+ abs=\S+ first=\S+ \S+ \S+ off=(-?\d+)")
REF = re.compile(r"(blk\s*-?\d+) l_out\s+abs=\s*\S+ first=\S+ \S+ \S+ off=(-?\d+)")


def engine_stages(path, bin_path):
    """pos -> blk -> stage -> vector, from the dump lines and the file they point into."""
    out = {}
    raw = np.memmap(bin_path, dtype=np.float32, mode="r") if bin_path else None
    for line in open(path, errors="ignore"):
        m = LINE.match(line)
        if not m:
            continue
        pos, blk, name, n, off = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
        if raw is None or off < 0:
            continue
        out.setdefault(pos, {})[(blk, name)] = np.asarray(raw[off:off + n], dtype=np.float64)
    return out


def ref_stages(path, bin_path):
    """blk -> vector, for the reference's l_out lines."""
    out = {}
    raw = np.memmap(bin_path, dtype=np.float32, mode="r") if bin_path else None
    for line in open(path, errors="ignore"):
        m = REF.match(line)
        if not m:
            continue
        label, off = m.group(1), int(m.group(2))
        if raw is None or off < 0:
            continue
        blk = int(label.split()[1])
        out[blk] = np.asarray(raw[off:off + N_EMBD_REF], dtype=np.float64)
    return out


N_EMBD_REF = 2304


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))


def main():
    eng_txt, eng_bin, ref_txt, ref_bin = sys.argv[1:5]
    eng = engine_stages(eng_txt, eng_bin)
    ref = ref_stages(ref_txt, ref_bin)
    if not eng:
        raise SystemExit("no engine stages: was STRATA_DUMP_BIN set for the run?")

    # the reference covers blk.0 .. blk.5 at position 0: the engine's pass 0 of that position
    pos0 = eng.get(0, {})
    rows, worst = [], None
    for blk in sorted(ref):
        got = pos0.get((blk, "l_out"))
        if got is None:
            print(f"blk {blk:2d} l_out  missing from the engine's dump")
            continue
        c = cos(got, ref[blk])
        rows.append((blk, c, float(np.abs(got - ref[blk]).max()), len(got)))
        if worst is None or c < worst[1]:
            worst = rows[-1]

    print(f"{'blk':>4} {'cosine':>12} {'max|d|':>12} {'n':>6}")
    for blk, c, d, n in rows:
        flag = "" if c >= 0.999 else "   <-- under 0.999"
        print(f"{blk:4d} {c:12.9f} {d:12.6f} {n:6d}{flag}")

    if worst is None:
        raise SystemExit("no stage in common")
    print(f"\nworst: blk {worst[0]} cosine {worst[1]:.9f}")
    print("gate (cosine >= 0.999 everywhere):", "PASS" if worst[1] >= 0.999 else "FAIL")
    raise SystemExit(0 if worst[1] >= 0.999 else 1)


if __name__ == "__main__":
    main()
