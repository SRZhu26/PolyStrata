"""Rewrite Inkling 2 Mini's Unsloth GGUF in the `inkling2_loop` layout mainline llama.cpp reads.

`unsloth/Inkling-2-Mini-GGUF` publishes the same model twice.  The root files use Unsloth's packed
loop layout; `Mainline_Support/*.gguf_file` uses the `inkling2_loop` layout, which is the one
`src/poly/inkling2.cpp` implements and the one `docs/models/reference/inkling2-loop.cpp` names.

Only the loop's message queries differ between the two.  The root file keeps them in two model-level
tensors, `loop_kv_res_query` and `loop_res_query`, of `[n_embd, 24]` each; the mainline file keeps one
`blk.N.loop_msg_query` of `[n_embd, 4]` per core layer.  Everything else is the same bytes at the
same length, so this copies it out of the local root file and takes only the message queries from the
mirror.  The root file also renames the loop's controllers (`loop_attn_*` and `loop_mlp_*` where the
mainline file has `loop_q_ctrl_*` and `loop_ffn_ctrl_*`); the bytes are equal.

Inputs, all small except the root file itself:

    ROOT.gguf             the root UD-Q4_K_XL from the mirror, 16.42 GiB, already local
    MAINLINE_HEAD.bin      the mirror's Mainline_Support file's first 64 MiB: its header
    MSG_QUERY_BLOCK.bin    the mirror's last 442,368 B: twelve `[n_embd, 4]` message queries

Evidence this rests on, recorded here so the result is checkable:

  * the mainline header's tensor offsets tile the file exactly: data_start + sum(bytes) == 17,629,760,736,
    every tensor contiguous, no encoding without a known block geometry;
  * 20 tensors sampled across the whole file compare byte-identical between the two layouts (Q8_0, NVFP4,
    F32 and BF16), which is the claim the model card makes;
  * the root file's `loop_kv_res_query` holds each core layer's k and v message queries at columns
    2*(il - core_start) and 2*(il - core_start) + 1, which compare byte-identical to the mainline file's.

usage: python tools/relayout_inkling2.py ROOT.gguf MAINLINE_HEAD.bin MSG_QUERY_BLOCK.bin OUT.gguf
"""
from __future__ import annotations

import io
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gguf_reader import GGUFFile  # noqa: E402

CHUNK = 8 << 20   # copy this much at a time, so the 16.42 GiB file costs no more memory than a tensor

# the release renamed the loop's controllers for the stream each one controls
ML_TO_ROOT = re.compile(r"^blk\.(\d+)\.loop_(q|k|v|ffn)_ctrl_(down|gain|forget)\.weight$")
CTRL = {"q": "attn", "k": "k", "v": "v", "ffn": "mlp"}
MSG_QUERY = re.compile(r"^blk\.(\d+)\.loop_msg_query\.weight$")
N_CTRL = 4   # q, k, v and ffn, the order the reference numbers them


def mainline_to_root(name: str) -> str:
    m = ML_TO_ROOT.match(name)
    if not m:
        return name
    return f"blk.{m.group(1)}.loop_{CTRL[m.group(2)]}_{m.group(3)}.weight"


def parse_header(head: pathlib.Path) -> GGUFFile:
    """Parse the mirror's header out of the file's first bytes, which is all a header read touches."""
    gguf = object.__new__(GGUFFile)
    gguf.metadata, gguf.tensors, gguf.version, gguf.alignment = {}, [], 0, 32
    gguf._parse(io.BytesIO(head.read_bytes()))
    return gguf


def copy_range(dst, src, length: int) -> None:
    left = length
    while left:
        buf = src.read(min(CHUNK, left))
        if not buf:
            raise RuntimeError("the source ended early")
        dst.write(buf)
        left -= len(buf)


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root_path, head_path, mq_path, out_path = (pathlib.Path(p) for p in sys.argv[1:])

    ml = parse_header(head_path)
    root = GGUFFile(root_path)

    # the header must be the one that describes the file: its tensors have to tile it with no gaps
    unknown = [t.name for t in ml.tensors if t.expected_bytes() is None]
    if unknown:
        print(f"relayout_inkling2: no block geometry for {unknown[:4]}", file=sys.stderr)
        return 1
    end = ml._data_start + sum(t.expected_bytes() for t in ml.tensors)
    print(f"relayout_inkling2: {len(ml.tensors)} tensors, data at {ml._data_start}, end {end}")

    # every mainline tensor has to find its bytes in one of the two sources
    root_at = {t.name: t for t in root.tensors}
    core_start = int(ml.metadata["inkling2_loop.loop.start"])
    n_embd = int(ml.metadata["inkling2_loop.embedding_length"])
    col = n_embd * 4   # a message query's column, in bytes
    sources: dict[str, tuple[str, int, int]] = {}   # name -> (which file, its offset, bytes)
    for t in ml.tensors:
        n = t.expected_bytes()
        m = MSG_QUERY.match(t.name)
        if m:
            # the mirror's last twelve tensors: four columns per core layer, in layer order
            sources[t.name] = ("mq", (int(m.group(1)) - core_start) * N_CTRL * col, n)
            continue
        was = mainline_to_root(t.name)
        if was not in root_at:
            print(f"relayout_inkling2: the root file has no {was} for {t.name}", file=sys.stderr)
            return 1
        if root_at[was].expected_bytes() != n:
            print(f"relayout_inkling2: {was} is {root_at[was].expected_bytes()} B, {t.name} is {n} B",
                  file=sys.stderr)
            return 1
        sources[t.name] = ("root", root.data_start + root_at[was].offset, n)
    n_mq = sum(1 for s in sources.values() if s[0] == "mq")
    print(f"relayout_inkling2: {len(sources)} tensors, {n_mq} message queries from the mirror")

    header = head_path.read_bytes()[: ml._data_start]
    with open(root_path, "rb") as src, open(mq_path, "rb") as mq, open(out_path, "wb") as dst:
        dst.write(header)
        # each tensor's place in the mirror's file, which is what its offsets say
        for t in sorted(ml.tensors, key=lambda t: t.offset):
            dst.seek(ml._data_start + t.offset)
            which, at, n = sources[t.name]
            if which == "mq":
                mq.seek(at)
                copy_range(dst, mq, n)
            else:
                src.seek(at)
                copy_range(dst, src, n)
        dst.truncate(end)

    print(f"relayout_inkling2: wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
