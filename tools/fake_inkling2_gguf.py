#!/usr/bin/env python3
"""Write a tiny GGUF whose metadata is Inkling 2 Mini's (inkling2_loop), with one small token_embd.weight.

Enough to take strata-poly's Engine::open() through Inkling2::load_geometry() (48 logical steps, KV row sizes) and up to
the first Inkling2::load_layer() bringup stub, with no 16 GB download.  Not a model: it cannot run a forward pass.
"""
import struct
import sys
from pathlib import Path

U32, F32, BOOL, STR, ARRAY = 4, 6, 7, 8, 9
ALIGN = 32


def s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k, v):
    return s(k) + struct.pack("<I", U32) + struct.pack("<I", v)


def kv_f32(k, v):
    return s(k) + struct.pack("<I", F32) + struct.pack("<f", v)


def kv_bool(k, v):
    return s(k) + struct.pack("<I", BOOL) + struct.pack("<B", 1 if v else 0)


def kv_str(k, v):
    return s(k) + struct.pack("<I", STR) + s(v)


def kv_arr_u32(k, vals):
    return s(k) + struct.pack("<I", ARRAY) + struct.pack("<I", U32) + struct.pack("<Q", len(vals)) + \
        b"".join(struct.pack("<I", v) for v in vals)


def kv_arr_bool(k, vals):
    return s(k) + struct.pack("<I", ARRAY) + struct.pack("<I", BOOL) + struct.pack("<Q", len(vals)) + \
        b"".join(struct.pack("<B", 1 if v else 0) for v in vals)


def main(out: Path) -> None:
    a = "inkling2_loop."
    n_layer, n_embd, n_vocab = 24, 2304, 8
    swa = [True] * n_layer
    for i in (5, 11, 17, 23):
        swa[i] = False
    kvs = [
        kv_str("general.architecture", "inkling2_loop"),
        kv_str("general.name", "synthetic-inkling2-geometry"),
        kv_u32(a + "block_count", n_layer),
        kv_u32(a + "embedding_length", n_embd),
        kv_u32(a + "vocab_size", n_vocab),
        kv_u32(a + "context_length", 1048576),
        kv_f32(a + "attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32(a + "expert_count", 128),
        kv_u32(a + "expert_used_count", 8),
        kv_u32(a + "expert_feed_forward_length", 1152),
        kv_arr_u32(a + "attention.head_count", [24] * n_layer),
        kv_arr_u32(a + "attention.head_count_kv", [8] * n_layer),
        kv_u32(a + "attention.key_length", 128),
        kv_u32(a + "attention.value_length", 128),
        kv_u32(a + "attention.sliding_window", 1024),
        kv_arr_bool(a + "attention.sliding_window_pattern", swa),
        kv_u32(a + "attention.relative_length", 16),
        kv_u32(a + "attention.relative_extent", 1024),
        kv_f32(a + "attention.qk_clip", 22.6274166),
        kv_f32(a + "attention.relative_clip", 8.0),
        kv_u32(a + "attention.log_scale_floor", 8192),
        kv_f32(a + "attention.log_scale_alpha", 0.1),
        kv_u32(a + "shortconv.l_cache", 4),
        kv_u32(a + "attn_res.block_size", 6),
        kv_f32(a + "activation.situ_beta", 4.0),
        kv_f32(a + "activation.situ_linear_beta", 25.0),
        kv_f32(a + "expert_weights_scale", 8.0),
        kv_f32(a + "final_logit_softcapping", 30.0),
        kv_u32(a + "loop.start", 6),
        kv_u32(a + "loop.end", 18),
        kv_u32(a + "loop.count", 3),
        kv_u32(a + "loop.gate_rank", 64),
    ]
    tensors = [("token_embd.weight", [n_embd, n_vocab], 0, 0)]   # F32, offset 0 from the data section
    n_bytes = n_embd * n_vocab * 4
    head = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs))
    meta = b"".join(kvs)
    info = b"".join(s(n) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", x) for x in ne) +
                    struct.pack("<I", t) + struct.pack("<Q", off) for n, ne, t, off in tensors)
    body = head + meta + info
    pad = (-len(body)) % ALIGN
    data = bytes(n_bytes)
    out.write_bytes(body + b"\0" * pad + data)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(kvs)} metadata keys, {len(tensors)} tensor)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "inkling2-geometry.gguf"))
