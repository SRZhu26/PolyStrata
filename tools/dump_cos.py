#!/usr/bin/env python3
"""The cosine similarity of the engine's dumps to an FP64 reference's, stage by stage.

The engine writes a line per dump under `--dump-layers`, and with `STRATA_DUMP_BIN` set it appends that
dump's floats to a file and puts where they begin on the line (`off=`).  `tools/ref_core.py` writes the
same for its own vectors under `STRATA_REF_BIN`, so a stage both name has can be compared in full, not just by
its sum, its absolute sum and its first three values.

    STRATA_DUMP_BIN=eng.f32 build-poly/strata-poly.exe ... --tokens 1000 --dump-layers 2> eng.txt
    STRATA_REF_BIN=ref.f32 python tools/ref_core.py FILE.gguf 1000 > ref.txt
    python tools/dump_cos.py eng.txt eng.f32 ref.txt ref.f32

A stage is matched on its name and its layer: the engine's `l_out` at blk.5 is the reference's `l_out` at blk.5.
A core layer runs three times; both sides keep the last visit, which is the one the epilogue sees.  The final
logits are compared too, past the vocabulary pad the reference drops.  The plan's gate is a cosine of at least
0.999 for every stage, and 0.999 at blk.6, blk.23 and the logits in particular.
"""
import re
import sys

import numpy as np

LINE = re.compile(r"dump pos=(-?\d+) blk=(-?\d+) (\S+)\s+n=(\d+) sum=\S+ abs=\S+ first=\S+ \S+ \S+ off=(-?\d+)")
REF = re.compile(r"(blk\s*-?\d+) l_out\s+abs=\s*\S+ first=\S+ \S+ \S+ off=(-?\d+)")
REF_LOGITS = re.compile(r"logits\s+abs=\s*\S+ first=\S+ \S+ \S+ off=(-?\d+)")

N_EMBD_REF = 2304
N_VOCAB_REF = 200058       # the unpadded vocabulary: output.weight has 201,024 rows, 966 of them pad
GATE = 0.999
GATE_BLKS = (6, 23)


def engine_stages(path, bin_path):
    """pos -> (blk, stage) -> vector, from the dump lines and the file they point into."""
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
    """blk -> vector for the reference's l_out lines, and 'logits' -> the output's."""
    out = {}
    raw = np.memmap(bin_path, dtype=np.float32, mode="r") if bin_path else None
    for line in open(path, errors="ignore"):
        m = REF.match(line)
        if m:
            blk, off = int(m.group(1).split()[1]), int(m.group(2))
            if raw is not None and off >= 0:
                out[blk] = np.asarray(raw[off:off + N_EMBD_REF], dtype=np.float64)
            continue
        m = REF_LOGITS.match(line)
        if m:
            off = int(m.group(1))
            if raw is not None and off >= 0:
                out["logits"] = np.asarray(raw[off:off + N_VOCAB_REF], dtype=np.float64)
    return out


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

    # both sides run position 0: the engine's first pass over it
    pos0 = eng.get(0, {})
    rows, worst = [], None
    for blk in sorted(k for k in ref if isinstance(k, int)):
        got = pos0.get((blk, "l_out"))
        if got is None:
            print(f"blk {blk:2d} l_out  missing from the engine's dump")
            continue
        c = cos(got, ref[blk])
        rows.append((f"blk {blk:2d}", c, float(np.abs(got - ref[blk]).max()), len(got)))
        if worst is None or c < worst[1]:
            worst = rows[-1]

    if "logits" in ref:
        got = pos0.get((24, "logits"))
        if got is None:
            got = pos0.get((23, "logits"))
        if got is None:
            print("logits      missing from the engine's dump")
        elif len(got) != len(ref["logits"]):
            print(f"logits      size {len(got)} in the engine, {len(ref['logits'])} in the reference")
        else:
            c = cos(got, ref["logits"])
            rows.append(("logits", c, float(np.abs(got - ref["logits"]).max()), len(got)))
            if worst is None or c < worst[1]:
                worst = rows[-1]

    print(f"{'stage':>9} {'cosine':>12} {'max|d|':>12} {'n':>7}")
    for label, c, d, n in rows:
        flag = "" if c >= GATE else "   <-- under 0.999"
        print(f"{label:>9} {c:12.9f} {d:12.6f} {n:7d}{flag}")

    if worst is None:
        raise SystemExit("no stage in common")
    # the plan names these three; say so separately, because a single worst number hides which
    named = {label: c for label, c, _, _ in rows if label in ("blk  6", "blk 23", "logits")}
    missing = [label for label in ("blk  6", "blk 23", "logits") if label not in named]
    for label in ("blk  6", "blk 23", "logits"):
        if label in named:
            print(f"gate point {label.strip():>7}: {named[label]:.9f}")
        else:
            print(f"gate point {label.strip():>7}: not compared")
    print(f"\nworst: {worst[0].strip()} cosine {worst[1]:.9f}")
    ok = worst[1] >= GATE and not missing and all(c >= GATE for c in named.values())
    print("gate (cosine >= 0.999 everywhere, and at blk.6, blk.23 and the logits):",
          "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
