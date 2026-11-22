#!/usr/bin/env python3
"""Write a complete tiny GGUF of architecture "inkling2_loop": every tensor Inkling2::load_layer() asks for.

The metadata is Inkling 2 Mini's (24 physical layers, a core looped three times, 1,024-token windows,
relative positions, the short convolutions) and the tensors are the ones that geometry needs, at dimensions
small enough to run on one card.  The values are noise: this is not a model, it only exercises the engine's
whole forward path - the schedule, the per-visit caches, every kernel - without the 17 GB download.

    python tools/fake_inkling2_full.py out.gguf
"""
import struct
import sys
from pathlib import Path

U32, F32, BOOL, STR, ARRAY = 4, 6, 7, 8, 9
F32_T, F16_T, Q8_0 = 0, 1, 8
ALIGN = 32

# Small enough to run on one card, wide enough that every dimension is a whole number of Q8_0 blocks.
N_LAYER, N_EMBD, N_VOCAB, N_UNPAD = 24, 256, 64, 48
N_HEAD, N_KV, HEAD_DIM = 8, 2, 32
D_REL, REL_EXTENT, WINDOW = 16, 64, 64
SCONV, BLOCK = 4, 6
N_EXPERT, N_USED, N_FF = 8, 2, 64
CORE_START, CORE_END, LOOPS, RANK = 6, 18, 3, 64
# the layers that attend the whole context rather than a window, as llama.cpp's is_swa() reads them
FULL = {5, 11, 17, 23}

CTRL = ("q", "k", "v", "ffn")


def s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k, v):
    return s(k) + struct.pack("<I", U32) + struct.pack("<I", v)


def kv_f32(k, v):
    return s(k) + struct.pack("<I", F32) + struct.pack("<f", v)


def kv_str(k, v):
    return s(k) + struct.pack("<I", STR) + s(v)


def kv_arr_u32(k, vals):
    return s(k) + struct.pack("<I", ARRAY) + struct.pack("<I", U32) + struct.pack("<Q", len(vals)) + \
        b"".join(struct.pack("<I", v) for v in vals)


def kv_arr_bool(k, vals):
    return s(k) + struct.pack("<I", ARRAY) + struct.pack("<I", BOOL) + struct.pack("<Q", len(vals)) + \
        b"".join(struct.pack("<B", 1 if v else 0) for v in vals)


def q8_0(n: int, seed: int) -> bytes:
    """`n` values of Q8_0: 34-byte blocks of an fp16 scale and 32 int8."""
    out = bytearray()
    x = seed & 0x7FFFFFFF or 1
    for _ in range(n // 32):
        vals = []
        for _ in range(32):
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            vals.append(((x >> 8) % 255 - 127) / 512.0)
        d = max(abs(v) for v in vals) / 127.0 or 1e-8
        out += struct.pack("<e", d)
        out += bytes((max(-127, min(127, round(v / d))) & 0xFF) for v in vals)
    return bytes(out)


def f32(n: int, seed: int, scale: float = 0.02) -> bytes:
    """`n` values centred on 0, and on 1 for a norm's gain."""
    out = bytearray()
    x = seed & 0x7FFFFFFF or 1
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out += struct.pack("<f", scale * (((x >> 8) % 2001) - 1000) / 1000.0)
    return bytes(out)


def tensors():
    """(name, dims, type, the data), dims fastest-varying first as GGUF stores them."""
    t = [
        ("token_embd.weight", [N_EMBD, N_VOCAB], F32_T, lambda: f32(N_EMBD * N_VOCAB, 7, 0.5)),
        ("token_embd_norm.weight", [N_EMBD], F32_T, lambda: f32(N_EMBD, 11, 1.0)),
        ("output.weight", [N_EMBD, N_VOCAB], F32_T, lambda: f32(N_EMBD * N_VOCAB, 13, 0.5)),
        ("output_norm.weight", [N_EMBD], F32_T, lambda: f32(N_EMBD, 17, 1.0)),
        ("output_res_query.weight", [N_EMBD], F32_T, lambda: f32(N_EMBD, 19, 0.3)),
    ]
    hq, hkv = N_HEAD * HEAD_DIM, N_KV * HEAD_DIM
    for il in range(N_LAYER):
        swa = il not in FULL
        p = "blk.%d." % il
        seed = 1000 + il * 97
        # the attention, its clip, and the relative-position projection and bias
        t += [
            (p + "attn_q.weight", [N_EMBD, hq], F32_T, lambda s=seed: f32(N_EMBD * hq, s, 0.02)),
            (p + "attn_k.weight", [N_EMBD, hkv], F32_T, lambda s=seed: f32(N_EMBD * hkv, s + 1, 0.02)),
            (p + "attn_v.weight", [N_EMBD, hkv], F32_T, lambda s=seed: f32(N_EMBD * hkv, s + 2, 0.02)),
            (p + "attn_output.weight", [hq, N_EMBD], F32_T, lambda s=seed: f32(hq * N_EMBD, s + 3, 0.02)),
            (p + "attn_gate.weight", [N_EMBD, hq], F32_T, lambda s=seed: f32(N_EMBD * hq, s + 4, 0.02)),
            (p + "attn_q_clip.weight", [HEAD_DIM, N_HEAD], F32_T, lambda s=seed: f32(HEAD_DIM * N_HEAD, s + 5, 1.0)),
            (p + "attn_rel.weight", [N_EMBD, D_REL * N_HEAD], F32_T, lambda s=seed: f32(N_EMBD * D_REL * N_HEAD, s + 6, 0.02)),
            (p + "attn_rel_clip.weight", [D_REL, N_HEAD], F32_T, lambda s=seed: f32(D_REL * N_HEAD, s + 7, 1.0)),
            (p + "attn_sink_proj.weight", [N_EMBD, N_HEAD], F32_T, lambda s=seed: f32(N_EMBD * N_HEAD, s + 8, 0.02)),
            (p + "attn_k_conv.weight", [SCONV, hkv], F32_T, lambda s=seed: f32(SCONV * hkv, s + 9, 0.3)),
            (p + "attn_v_conv.weight", [SCONV, hkv], F32_T, lambda s=seed: f32(SCONV * hkv, s + 10, 0.3)),
            (p + "attn_post_conv.weight", [SCONV, N_EMBD], F32_T, lambda s=seed: f32(SCONV * N_EMBD, s + 11, 0.3)),
            (p + "ffn_post_conv.weight", [SCONV, N_EMBD], F32_T, lambda s=seed: f32(SCONV * N_EMBD, s + 12, 0.3)),
            (p + "attn_rel_bias.weight", [D_REL, WINDOW if swa else REL_EXTENT], F32_T,
             lambda s=seed: f32(D_REL * (WINDOW if swa else REL_EXTENT), s + 13, 0.05)),
        ]
        if swa:
            t.append((p + "attn_norm.weight", [N_EMBD], F32_T, lambda s=seed: f32(N_EMBD, s + 14, 1.0)))
        else:
            t += [
                (p + "attn_q_inp_norm.weight", [N_EMBD], F32_T, lambda s=seed: f32(N_EMBD, s + 14, 1.0)),
                (p + "attn_k_inp_norm.weight", [N_EMBD], F32_T, lambda s=seed: f32(N_EMBD, s + 15, 1.0)),
                (p + "attn_v_inp_norm.weight", [N_EMBD], F32_T, lambda s=seed: f32(N_EMBD, s + 16, 1.0)),
                (p + "attn_k_conv_proj.weight", [N_EMBD, SCONV * N_KV], F32_T,
                 lambda s=seed: f32(N_EMBD * SCONV * N_KV, s + 17, 0.02)),
                (p + "attn_v_conv_proj.weight", [N_EMBD, SCONV * N_KV], F32_T,
                 lambda s=seed: f32(N_EMBD * SCONV * N_KV, s + 18, 0.02)),
            ]
        # the depth attention's queries: the first layer of a block starts a new stream and has none
        if il > 0:
            t.append((p + "attn_res_query.weight", [N_EMBD, 1 if swa else 3], F32_T,
                      lambda s=seed: f32(N_EMBD * (1 if swa else 3), s + 19, 0.3)))
        t += [
            (p + "ffn_res_query.weight", [N_EMBD], F32_T, lambda s=seed: f32(N_EMBD, s + 20, 0.3)),
            (p + "ffn_norm.weight", [N_EMBD], F32_T, lambda s=seed: f32(N_EMBD, s + 21, 1.0)),
            (p + "ffn_gate_inp.weight", [N_EMBD, N_EXPERT], F32_T, lambda s=seed: f32(N_EMBD * N_EXPERT, s + 22, 0.02)),
            (p + "exp_probs_b.bias", [N_EXPERT], F32_T, lambda s=seed: f32(N_EXPERT, s + 23, 0.1)),
            # the routed experts, gate and up fused as llama.cpp keeps them: one row of 2 * n_ff
            (p + "ffn_gate_up_exps.weight", [N_EMBD, 2 * N_FF, N_EXPERT], Q8_0,
             lambda s=seed: q8_0(N_EMBD * 2 * N_FF * N_EXPERT, s + 24)),
            (p + "ffn_down_exps.weight", [N_FF, N_EMBD, N_EXPERT], Q8_0,
             lambda s=seed: q8_0(N_FF * N_EMBD * N_EXPERT, s + 25)),
        ]
        # the looped core's four controllers, shared by every pass over the layer
        if CORE_START <= il < CORE_END:
            t.append((p + "loop_msg_query.weight", [N_EMBD, 4], F32_T, lambda s=seed: f32(N_EMBD * 4, s + 26, 0.3)))
            for c, name in enumerate(CTRL):
                t += [
                    (p + "loop_%s_ctrl_down.weight" % name, [2 * N_EMBD, RANK], F32_T,
                     lambda s=seed, c=c: f32(2 * N_EMBD * RANK, s + 27 + c, 0.02)),
                    (p + "loop_%s_ctrl_gain.weight" % name, [RANK, N_EMBD], F32_T,
                     lambda s=seed, c=c: f32(RANK * N_EMBD, s + 31 + c, 0.02)),
                    (p + "loop_%s_ctrl_forget.weight" % name, [RANK, N_EMBD], F32_T,
                     lambda s=seed, c=c: f32(RANK * N_EMBD, s + 35 + c, 0.02)),
                ]
    return t


def main(out: Path) -> None:
    a = "inkling2_loop."
    swa = [il not in FULL for il in range(N_LAYER)]
    kvs = [
        kv_str("general.architecture", "inkling2_loop"),
        kv_str("general.name", "synthetic-inkling2"),
        kv_u32("general.alignment", ALIGN),
        kv_u32(a + "block_count", N_LAYER),
        kv_u32(a + "embedding_length", N_EMBD),
        kv_u32(a + "vocab_size", N_VOCAB),
        kv_u32(a + "vocab_size_unpadded", N_UNPAD),
        kv_u32(a + "context_length", 1 << 20),
        kv_f32(a + "attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32(a + "expert_count", N_EXPERT),
        kv_u32(a + "expert_used_count", N_USED),
        kv_u32(a + "expert_feed_forward_length", N_FF),
        kv_arr_u32(a + "attention.head_count", [N_HEAD] * N_LAYER),
        kv_arr_u32(a + "attention.head_count_kv", [N_KV] * N_LAYER),
        kv_u32(a + "attention.key_length", HEAD_DIM),
        kv_u32(a + "attention.value_length", HEAD_DIM),
        kv_u32(a + "attention.sliding_window", WINDOW),
        kv_arr_bool(a + "attention.sliding_window_pattern", swa),
        kv_u32(a + "attention.relative_length", D_REL),
        kv_u32(a + "attention.relative_extent", REL_EXTENT),
        kv_f32(a + "attention.qk_clip", 22.6274166),
        kv_f32(a + "attention.relative_clip", 8.0),
        kv_u32(a + "attention.log_scale_floor", 8192),
        kv_f32(a + "attention.log_scale_alpha", 0.1),
        kv_u32(a + "shortconv.l_cache", SCONV),
        kv_u32(a + "attn_res.block_size", BLOCK),
        kv_f32(a + "activation.situ_beta", 4.0),
        kv_f32(a + "activation.situ_linear_beta", 25.0),
        kv_f32(a + "expert_weights_scale", 8.0),
        kv_f32(a + "final_logit_softcapping", 30.0),
        kv_u32(a + "loop.start", CORE_START),
        kv_u32(a + "loop.end", CORE_END),
        kv_u32(a + "loop.count", LOOPS),
        kv_u32(a + "loop.gate_rank", RANK),
    ]

    # Every tensor's data, in the order the infos list them, each starting a 32-byte boundary.
    infos, blobs, at = [], [], 0
    for name, dims, type, make in tensors():
        data = make()
        blobs.append(data)
        infos.append(s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) +
                      struct.pack("<I", type) + struct.pack("<Q", at))
        at += (len(data) + ALIGN - 1) // ALIGN * ALIGN

    head = struct.pack("<IIQQ", 0x46554747, 3, len(infos), len(kvs))
    body = head + b"".join(kvs) + b"".join(infos)
    pad = (-len(body)) % ALIGN
    data = bytearray()
    for blob in blobs:
        data += blob + bytes((-len(blob)) % ALIGN)
    out.write_bytes(body + bytes(pad) + bytes(data))
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(kvs)} metadata keys, {len(infos)} tensors)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "inkling2-full.gguf"))
