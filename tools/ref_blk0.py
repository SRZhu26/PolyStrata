#!/usr/bin/env python3
"""An FP64 reference for blk.0's attention at position 0, to compare against `--dump-layers`.

blk.0 is the simplest visit of the schedule: a window layer (so one depth mix, no messages, no
log-scaled position), the first layer of the first block (so it starts a residual stream and has no
attn_res_query), and the first token of a call (so every convolution reads a zero state and only the
key at distance 0 exists).

    python tools/ref_blk0.py FILE.gguf TOKEN
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gguf_reader import GGUFFile

N_EMBD, N_HEAD, N_KV, HEAD_DIM = 2304, 24, 8, 128
D_REL, REL_EXTENT, K = 16, 1024, 4
QK_CLIP, REL_CLIP, EPS = 22.627416610717773, 8.0, 1e-6
N_EXPERT, N_USED, N_FF = 128, 8, 1152
W_SCALE, SITU_BETA, SITU_LB = 8.0, 4.0, 25.0


# e2m1's values doubled, as ggml's kvalues_fp4 has them; the block scale is halved to match
KV_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float64)


def ue4m3_table():
    """ggml_ue4m3_to_fp32 over all 256 codes: 1 sign-free exp of 4 bits (bias 7), 3 mantissa bits."""
    t = np.zeros(256)
    for x in range(256):
        if x == 0 or x == 0x7F:
            continue
        e, m = (x >> 3) & 0xF, x & 0x7
        t[x] = (m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)) * 0.5
    return t


UE4M3 = ue4m3_table()


def dequant_nvfp4(g, t, expert, nrow, ncol):
    """One expert's [ncol, nrow] slab of an NVFP4 tensor, as fp64.

    A block is 64 values: four E4M3 sub-block scales of 16 values each, then 32 bytes of packed
    E2M1 nibbles whose low halves are the sub-block's first 8 values and high halves its last 8.
    """
    nb = ncol // 64
    row_bytes = nb * 36
    with open(g.path, "rb") as fh:
        fh.seek(g.data_start + t.offset + expert * nrow * row_bytes)
        raw = np.frombuffer(fh.read(nrow * row_bytes), dtype=np.uint8).reshape(nrow, nb, 36)
    d = UE4M3[raw[:, :, 0:4]]          # (nrow, nb, 4)
    qs = raw[:, :, 4:36]                # (nrow, nb, 32)
    out = np.empty((nrow, nb, 64))
    for s in range(4):
        blk = qs[:, :, s * 8:(s + 1) * 8]
        out[:, :, s * 16:s * 16 + 8] = KV_FP4[blk & 0x0F]
        out[:, :, s * 16 + 8:s * 16 + 16] = KV_FP4[blk >> 4]
    return (out.reshape(nrow, nb, 4, 16) * d[:, :, :, None]).reshape(nrow, ncol)


def expert_scale(g, name, expert):
    """A converted expert matrix's global scale, "<base>.scale" beside "<base>.weight", one float per expert."""
    t = {x.name: x for x in g.tensors}[name + ".scale"]
    with open(g.path, "rb") as fh:
        fh.seek(g.data_start + t.offset + 4 * expert)
        return float(np.frombuffer(fh.read(4), dtype=np.float32)[0])


def situ(gate, up):
    return SITU_BETA * np.tanh(gate / SITU_BETA) / (1.0 + np.exp(-gate)) * SITU_LB * np.tanh(up / SITU_LB)


def mix(states, query):
    """The depth attention: rms_norm every state, softmax over the scores, then their weighted sum."""
    S = np.stack([rms(s) for s in states])
    scores = S @ query / np.sqrt(N_EMBD)
    p = np.exp(scores - scores.max())
    return (p / p.sum()) @ S


def dequant_q8_0(t, g, row, ncol):
    nb = ncol // 32
    with open(g.path, "rb") as fh:
        fh.seek(g.data_start + t.offset + row * nb * 34)
        raw = fh.read(nb * 34)
    out = np.empty(ncol, dtype=np.float64)
    for b in range(nb):
        d = np.frombuffer(raw[b * 34:b * 34 + 2], dtype=np.float16)[0].astype(np.float64)
        q = np.frombuffer(raw[b * 34 + 2:b * 34 + 34], dtype=np.int8).astype(np.float64)
        out[b * 32:(b + 1) * 32] = d * q
    return out


def load(g, name):
    """A 2-D tensor of the file as fp64, ne0 the fastest, i.e. the row of ggml's layout."""
    t = {x.name: x for x in g.tensors}[name]
    n0 = t.shape[0]
    n1 = t.elements // n0
    if t.type_name == "F32":
        with open(g.path, "rb") as fh:
            fh.seek(g.data_start + t.offset)
            v = np.frombuffer(fh.read(t.elements * 4), dtype=np.float32)
    elif t.type_name == "BF16":
        with open(g.path, "rb") as fh:
            fh.seek(g.data_start + t.offset)
            v = (np.frombuffer(fh.read(t.elements * 2), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif t.type_name == "Q8_0":
        v = np.stack([dequant_q8_0(t, g, r, n0) for r in range(n1)])
    else:
        raise SystemExit(f"{name}: {t.type_name}")
    return v.astype(np.float64).reshape(n1, n0)


def rms(x, eps=EPS):
    return x / np.sqrt(np.mean(x * x) + eps)


def clip(x, scale, dim, heads, threshold):
    y = x.copy()
    if scale is not None:
        y = y * scale.reshape(heads, dim).reshape(-1)
    for h in range(heads):
        v = y[h * dim:(h + 1) * dim]
        n = np.sqrt(np.sum(v * v)) + 1e-6
        y[h * dim:(h + 1) * dim] = v * min(max(threshold / n, 0.0), 1.0)
    return y


def conv(x, state, w):
    """out[t][c] = (x[t][c] + sum_i w[c][i] x_ext[t+i][c]) / sqrt(k+1), static taps."""
    ch = x.shape[1]
    ext = np.concatenate([state, x], axis=0)
    out = np.empty_like(x)
    for c in range(ch):
        acc = np.zeros(x.shape[0])
        for i in range(K):
            acc += w[c, i] * ext[3 - (K - 1) + i:3 - (K - 1) + i + x.shape[0], c]
        out[:, c] = (x[:, c] + acc) / np.sqrt(K + 1.0)
    return out


def main():
    path = pathlib.Path(sys.argv[1])
    tok = int(sys.argv[2])
    g = GGUFFile(path)
    W = lambda n: load(g, n)
    V = lambda n: load(g, n).ravel()
    p = "blk.0."

    # the normed embedding, which the dump's `tok` line already confirms
    row = dequant_q8_0({x.name: x for x in g.tensors}["token_embd.weight"], g, tok, N_EMBD)
    r = rms(row) * V("token_embd_norm.weight")
    print(f"tok      abs={np.abs(r).sum():12.6f} first={r[0]:.6f} {r[1]:.6f} {r[2]:.6f}")

    # a window layer, first layer of a block, no messages: the depth mix is `cur` alone
    xin = rms(r * 1.0, EPS) if False else rms(r, EPS) * V(p + "attn_norm.weight")
    print(f"attn_in  abs={np.abs(xin).sum():12.6f} first={xin[0]:.6f} {xin[1]:.6f} {xin[2]:.6f}")

    qraw = W(p + "attn_q.weight") @ xin
    kraw = W(p + "attn_k.weight") @ xin
    vraw = W(p + "attn_v.weight") @ xin
    rraw = W(p + "attn_rel.weight") @ xin

    zero = np.zeros((K - 1, N_KV * HEAD_DIM))
    k = conv(kraw.reshape(1, -1), zero, W(p + "attn_k_conv.weight"))[0]
    v = conv(vraw.reshape(1, -1), zero, W(p + "attn_v_conv.weight"))[0]

    q = clip(qraw, W(p + "attn_q_clip.weight"), HEAD_DIM, N_HEAD, QK_CLIP)
    k = clip(k, None, HEAD_DIM, N_KV, QK_CLIP)
    vn = np.concatenate([rms(v[h * HEAD_DIM:(h + 1) * HEAD_DIM]) for h in range(N_KV)])
    rr = clip(rraw, W(p + "attn_rel_clip.weight"), D_REL, N_HEAD, REL_CLIP)

    sinks = W(p + "attn_sink_proj.weight") @ xin
    bias = W(p + "attn_rel_bias.weight")          # [extent, d_rel]
    # one key, at distance 0: kqb[d, h] = sum_r bias[0, r] rcur[h, r] - sink[h]
    kqb = np.array([bias[0] @ rr[h * D_REL:(h + 1) * D_REL] for h in range(N_HEAD)]) - sinks

    att = np.empty(N_HEAD * HEAD_DIM)
    for h in range(N_HEAD):
        hkv = h // (N_HEAD // N_KV)
        kh = k[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]
        logit = float(q[h * HEAD_DIM:(h + 1) * HEAD_DIM] @ kh) / np.sqrt(HEAD_DIM) + kqb[h]
        best = max(logit, 0.0)                    # the sink's own logit is zero
        w0 = np.exp(logit - best) / (np.exp(logit - best) + np.exp(-best))
        att[h * HEAD_DIM:(h + 1) * HEAD_DIM] = w0 * vn[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]

    print(f"q      abs={np.abs(q).sum():12.6f}  k={np.abs(k).sum():.4f} vn={np.abs(vn).sum():.4f} kqb={np.abs(kqb).sum():.4f}")
    print(f"r_clip abs={np.abs(rr).sum():12.6f}  sinks={np.abs(sinks).sum():12.6f}  gate_abs={np.abs(W(p + 'attn_gate.weight') @ xin).sum():12.6f}")
    for h in range(4):
        hkv = h // (N_HEAD // N_KV)
        kh = k[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]
        qk = float(q[h * HEAD_DIM:(h + 1) * HEAD_DIM] @ kh) / np.sqrt(HEAD_DIM)
        bt = float(bias[0] @ rr[h * D_REL:(h + 1) * D_REL])
        logit = qk + bt - sinks[h]
        best = max(logit, 0.0)
        w0 = np.exp(logit - best) / (np.exp(logit - best) + np.exp(-best))
        seg = att[h * HEAD_DIM:(h + 1) * HEAD_DIM]
        print(f"  h{h}: qk={qk:9.5f} bias={bt:9.5f} sink={sinks[h]:9.5f} logit={logit:9.5f} w0={w0:.6f} "
              f"|att|={np.abs(seg).sum():9.5f} |vn|={np.abs(vn[hkv*HEAD_DIM:(hkv+1)*HEAD_DIM]).sum():9.5f} "
              f"k0={k[hkv*HEAD_DIM]:.6f} vn0={vn[hkv*HEAD_DIM]:.6f}")
    print(f"att_pregate abs={np.abs(att).sum():12.6f}")
    gate = 1.0 / (1.0 + np.exp(-(W(p + "attn_gate.weight") @ xin)))
    print(f"gate   mean={gate.mean():.6f} min={gate.min():.6f} max={gate.max():.6f}")
    att *= gate
    print(f"att_postgate abs={np.abs(att).sum():12.6f}")
    conv_ = W(p + "attn_output.weight") @ att
    print(f"conv_  abs={np.abs(conv_).sum():12.6f}")
    # the same without the gate and without the V norm, to see which the engine's 451.11 matches
    att_pg = np.empty(N_HEAD * HEAD_DIM)
    for h in range(N_HEAD):
        hkv = h // (N_HEAD // N_KV)
        kh = k[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]
        logit = float(q[h * HEAD_DIM:(h + 1) * HEAD_DIM] @ kh) / np.sqrt(HEAD_DIM) + kqb[h]
        best = max(logit, 0.0)
        w0 = np.exp(logit - best) / (np.exp(logit - best) + np.exp(-best))
        att_pg[h * HEAD_DIM:(h + 1) * HEAD_DIM] = w0 * v[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]
    c2 = W(p + "attn_output.weight") @ att_pg
    a2 = conv(c2.reshape(1, -1), np.zeros((K - 1, N_EMBD)), W(p + "attn_post_conv.weight"))[0]
    print(f"no-vnorm attn abs={np.abs(a2).sum():12.6f}")
    c3 = W(p + "attn_output.weight") @ (att / gate)
    a3 = conv(c3.reshape(1, -1), np.zeros((K - 1, N_EMBD)), W(p + "attn_post_conv.weight"))[0]
    print(f"no-gate  attn abs={np.abs(a3).sum():12.6f}")
    c4 = W(p + "attn_output.weight") @ att_pg
    print(f"no-vnorm no-gate attn abs={np.abs(conv(c4.reshape(1,-1), np.zeros((K-1,N_EMBD)), W(p+'attn_post_conv.weight'))[0]).sum():12.6f}")
    # the gate read as one value per head (its head's first channel) instead of one per channel
    g1 = np.repeat(gate[::HEAD_DIM], HEAD_DIM)
    c5 = W(p + "attn_output.weight") @ (att * gate / g1)
    a5 = conv(c5.reshape(1, -1), np.zeros((K - 1, N_EMBD)), W(p + "attn_post_conv.weight"))[0]
    print(f"head-gate attn abs={np.abs(a5).sum():12.6f}  (engine's was 451.111689)")
    print(f"conv_+out abs={np.abs(conv_ + np.zeros(N_EMBD)).sum():.4f}  sqrt5*ao={(np.sqrt(5)*143.642527):.4f}")
    ao = conv(conv_.reshape(1, -1), np.zeros((K - 1, N_EMBD)), W(p + "attn_post_conv.weight"))[0]
    print(f"attn     abs={np.abs(ao).sum():12.6f} first={ao[0]:.6f} {ao[1]:.6f} {ao[2]:.6f}")

    # ---- the MoE.  blk.0 starts a block, so its input joined the bank and the residual is the
    # attention's output alone; the depth attention then mixes that with the block's inputs.
    ffn_in = mix([ao, r], V(p + "ffn_res_query.weight"))
    xn = rms(ffn_in) * V(p + "ffn_norm.weight")
    print(f"ffn_in   abs={np.abs(ffn_in).sum():12.6f} first={ffn_in[0]:.6f} {ffn_in[1]:.6f} {ffn_in[2]:.6f}")
    print(f"xn       abs={np.abs(xn).sum():12.6f} first={xn[0]:.6f} {xn[1]:.6f} {xn[2]:.6f}")
    print(f"rms(ao)  abs={np.abs(rms(ao)).sum():12.6f}  rms(r)={np.abs(rms(r)).sum():12.6f}")

    logits = W(p + "ffn_gate_inp.weight") @ xn
    probs = 1.0 / (1.0 + np.exp(-logits))
    # the selection is biased (DeepSeek V3's rule), the weights are not
    sel = np.argsort(-(probs + V(p + "exp_probs_b.bias")))[:N_USED]
    w = probs[sel] / probs[sel].sum() * W_SCALE
    print(f"router   top={sel.tolist()} w={np.round(w, 6).tolist()}")

    kgu, kdn = "ffn_gate_up_exps", "ffn_down_exps"
    tgu = {x.name: x for x in g.tensors}[p + kgu + ".weight"]
    tdn = {x.name: x for x in g.tensors}[p + kdn + ".weight"]
    # llama.cpp applies the router's weight to the experts' sum, not to their input (weight_before_ffn is
    # llama4's alone), so the gate and up products are of the plain FFN input
    tags = ["correct", "weight inside", "halves swapped", "silu", "no weights"]
    acc = {t: np.zeros(N_EMBD) for t in tags}
    for k, e in enumerate(sel):
        sgu = expert_scale(g, p + kgu, int(e))
        sdn = expert_scale(g, p + kdn, int(e))
        wgu = dequant_nvfp4(g, tgu, int(e), N_FF * 2, N_EMBD) * sgu
        wdn = dequant_nvfp4(g, tdn, int(e), N_EMBD, N_FF) * sdn
        gate, up = wgu[:N_FF] @ xn, wgu[N_FF:] @ xn
        print(f"  e{int(e):3d} sgu={sgu:.6e} sdn={sdn:.6e} |gate|={np.abs(gate).sum():9.4f} "
              f"|up|={np.abs(up).sum():9.4f} |situ|={np.abs(situ(gate, up)).sum():.6e}")
        acc["correct"] += w[k] * (wdn @ situ(gate, up))
        acc["weight inside"] += w[k] * (wdn @ situ(gate * w[k], up * w[k]))
        acc["halves swapped"] += w[k] * (wdn @ situ(up, gate))
        acc["silu"] += w[k] * (wdn @ (gate / (1.0 + np.exp(-gate)) * up))
        acc["no weights"] += wdn @ situ(gate, up)
        print(f"  e{int(e):3d} w={w[k]:.6f} |dn|={np.abs(wdn @ situ(gate, up)).sum():9.4f}")
    for t in tags:
        print(f"ffn {t:16s} abs={np.abs(acc[t]).sum():12.6f} first={acc[t][0]:.6f} {acc[t][1]:.6f} {acc[t][2]:.6f}")
        fn = conv(acc[t].reshape(1, -1), np.zeros((K - 1, N_EMBD)), W(p + "ffn_post_conv.weight"))[0]
        # the FFN's residual is the attention's output, not the mix its norm was taken of
        print(f"    ffn_out abs={np.abs(fn).sum():12.6f} l_out abs={np.abs(ao + fn).sum():12.6f} "
              f"first={ao[0] + fn[0]:.6f} {ao[1] + fn[1]:.6f} {ao[2] + fn[2]:.6f}")
    out = acc["correct"]
    fn = conv(out.reshape(1, -1), np.zeros((K - 1, N_EMBD)), W(p + "ffn_post_conv.weight"))[0]
    print(f"ffn_out  abs={np.abs(fn).sum():12.6f} first={fn[0]:.6f} {fn[1]:.6f} {fn[2]:.6f}")
    lo = ao + fn
    print(f"l_out    abs={np.abs(lo).sum():12.6f} first={lo[0]:.6f} {lo[1]:.6f} {lo[2]:.6f}")


if __name__ == "__main__":
    main()
