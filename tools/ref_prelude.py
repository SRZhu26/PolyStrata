#!/usr/bin/env python3
"""An FP64 reference for the prelude's layers (blk.0 .. blk.5) at position 0, one token.

blk.0 is a window layer that starts the first block; blk.1 .. blk.4 are window layers that
join it; blk.5 is the first full-context layer, so it takes three depth mixes, three input
norms, a dynamic short convolution and the log-scaled position.  Position 0 makes `tau` exactly
1, every convolution's state zero, and the attention a single key at distance 0.

Each layer's `l_out` is printed beside the engine's `--dump-layers` line of the same name.

    python tools/ref_prelude.py FILE.gguf TOKEN
"""
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gguf_reader import GGUFFile
from ref_blk0 import (EPS, HEAD_DIM, K, N_EMBD, N_FF, N_HEAD, N_KV, QK_CLIP, REL_CLIP, SITU_BETA, SITU_LB,
                      W_SCALE, N_EXPERT, N_USED, clip, conv, dequant_nvfp4, dequant_q8_0, expert_scale, load, rms,
                      situ)

D_REL = 16
GLOBAL_LAYERS = (5, 11, 17, 23)     # the full-context layers: the rest are window (SWA) layers
BLOCK = 6                            # the layers of a block, whose first one starts a residual stream
N_PRELUDE = 6


def mix_q(states, query):
    """The depth attention for `query` of shape [n_q, n_embd]: a softmax over the RMS-normed states."""
    S = np.stack([rms(s) for s in states])            # [n_states, n_embd]
    scores = S @ query.T / np.sqrt(N_EMBD)             # [n_states, n_q]
    out = np.empty((query.shape[0], N_EMBD))
    for q in range(query.shape[0]):
        sc = scores[:, q]
        p = np.exp(sc - sc.max())
        out[q] = (p / p.sum()) @ S
    return out


def conv_dyn(x, state, w, dyn=None):
    """The causal short convolution of one token, with `dyn`'s taps of shape [n_head, k] when given.

    out[c] = (x[c] + sum_i w[c, i] ext[t + i] + sum_p dyn[h, p] ext[k - 1 + t - p]) / sqrt((2 if dyn else 1)*k + 1)
    where ext = concat(state, x) along time and h = c / (channels / n_head).
    """
    ch = x.shape[1]
    ext = np.concatenate([state, x], axis=0)          # [k - 1 + n_tok, ch]
    out = np.empty_like(x)
    scale = 1.0 / np.sqrt((2.0 if dyn is not None else 1.0) * K + 1.0)
    per = ch // (dyn.shape[0] // K) if dyn is not None else 0
    for t in range(x.shape[0]):
        for c in range(ch):
            acc = 0.0
            for i in range(K):
                acc += w[c, i] * ext[t + i, c]
            if dyn is not None:
                h = c // per
                for p in range(K):
                    acc += dyn[h, p] * ext[K - 1 + t - p, c]
            out[t, c] = (x[t, c] + acc) * scale
    return out


def run_layer(g, il, cur, blocks, W, V, pos=0):
    """One layer from its input to `l_out`, as the reference's build_layer() does."""
    p = f"blk.{il}."
    swa = il not in GLOBAL_LAYERS
    ip = il % BLOCK
    n_q = 1 if swa else 3

    inpSA = cur
    inputs = [cur.copy(), cur.copy(), cur.copy()]
    if il > 0:                                          # a block's first layer has no query
        q = V(p + "attn_res_query.weight").reshape(n_q, N_EMBD)
        m = mix_q(blocks + [cur], q)
        inputs = [m[0].copy() for _ in range(3)] if n_q == 1 else [m[j].copy() for j in range(3)]
    # the three input norms: one for a window layer, one each for a full-context one
    if swa:
        n = rms(inputs[0]) * V(p + "attn_norm.weight")
        inputs = [n.copy(), n.copy(), n.copy()]
    else:
        inputs = [rms(inputs[0]) * V(p + "attn_q_inp_norm.weight"),
                  rms(inputs[1]) * V(p + "attn_k_inp_norm.weight"),
                  rms(inputs[2]) * V(p + "attn_v_inp_norm.weight")]

    Qraw = W(p + "attn_q.weight") @ inputs[0]
    Kraw = W(p + "attn_k.weight") @ inputs[1]
    Vraw = W(p + "attn_v.weight") @ inputs[2]
    Rraw = W(p + "attn_rel.weight") @ inputs[0]

    zero = np.zeros((K - 1, N_KV * HEAD_DIM))
    kdyn = None if swa else (W(p + "attn_k_conv_proj.weight") @ inputs[1]).reshape(K, N_KV)
    vdyn = None if swa else (W(p + "attn_v_conv_proj.weight") @ inputs[2]).reshape(K, N_KV)
    k = conv_dyn(Kraw.reshape(1, -1), zero, W(p + "attn_k_conv.weight"), kdyn)[0]
    v = conv_dyn(Vraw.reshape(1, -1), zero, W(p + "attn_v_conv.weight"), vdyn)[0]

    q = clip(Qraw, W(p + "attn_q_clip.weight"), HEAD_DIM, N_HEAD, QK_CLIP)
    k = clip(k, None, HEAD_DIM, N_KV, QK_CLIP)
    vn = np.concatenate([rms(v[h * HEAD_DIM:(h + 1) * HEAD_DIM]) for h in range(N_KV)])
    r = clip(Rraw, W(p + "attn_rel_clip.weight"), D_REL, N_HEAD, REL_CLIP)
    # the log-scaled position multiplies both the queries and the relative projection; 1 at pos 0
    tau = 1.0 + 0.1 * np.log(max(1.0, (pos + 1) / 8192.0))
    if not swa:
        q = q * tau
        r = r * tau

    sinks = W(p + "attn_sink_proj.weight") @ inputs[0]
    bias = W(p + "attn_rel_bias.weight")              # [extent, d_rel]
    kqb = np.array([bias[0] @ r[h * D_REL:(h + 1) * D_REL] for h in range(N_HEAD)]) - sinks

    att = np.empty(N_HEAD * HEAD_DIM)
    for h in range(N_HEAD):
        hkv = h // (N_HEAD // N_KV)
        kh = k[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]
        logit = float(q[h * HEAD_DIM:(h + 1) * HEAD_DIM] @ kh) / np.sqrt(HEAD_DIM) + kqb[h]
        best = max(logit, 0.0)
        w0 = np.exp(logit - best) / (np.exp(logit - best) + np.exp(-best))
        att[h * HEAD_DIM:(h + 1) * HEAD_DIM] = w0 * vn[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]

    gate = 1.0 / (1.0 + np.exp(-(W(p + "attn_gate.weight") @ inputs[0])))
    att = att * gate
    ao = conv_dyn((W(p + "attn_output.weight") @ att).reshape(1, -1), np.zeros((K - 1, N_EMBD)),
                  W(p + "attn_post_conv.weight"), None)[0]

    # a block's first layer starts a new residual stream, the others add to the one they found
    if ip == 0:
        blocks.append(inpSA)
        cur = ao.copy()
    else:
        cur = inpSA + ao
    ffn_inp = cur.copy()

    m = mix_q(blocks + [cur], V(p + "ffn_res_query.weight").reshape(1, N_EMBD))
    xn = rms(m[0]) * V(p + "ffn_norm.weight")

    logits = W(p + "ffn_gate_inp.weight") @ xn
    probs = 1.0 / (1.0 + np.exp(-logits))
    sel = np.argsort(-(probs + V(p + "exp_probs_b.bias")))[:N_USED]
    w = probs[sel] / probs[sel].sum() * W_SCALE
    tens = {x.name: x for x in g.tensors}
    acc = np.zeros(N_EMBD)
    for kk, e in enumerate(sel):
        wgu = dequant_nvfp4(g, tens[p + "ffn_gate_up_exps.weight"], int(e), N_FF * 2, N_EMBD) * \
            expert_scale(g, p + "ffn_gate_up_exps", int(e))
        wdn = dequant_nvfp4(g, tens[p + "ffn_down_exps.weight"], int(e), N_EMBD, N_FF) * \
            expert_scale(g, p + "ffn_down_exps", int(e))
        acc += w[kk] * (wdn @ situ(wgu[:N_FF] @ xn, wgu[N_FF:] @ xn))
    fn = conv_dyn(acc.reshape(1, -1), np.zeros((K - 1, N_EMBD)), W(p + "ffn_post_conv.weight"), None)[0]
    return ffn_inp + fn


def main():
    path = pathlib.Path(sys.argv[1])
    tok = int(sys.argv[2])
    g = GGUFFile(path)
    W = lambda n: load(g, n)
    V = lambda n: load(g, n).ravel()

    # STRATA_REF_BIN: the vectors themselves beside the summary, for tools/dump_cos.py
    bin_path = os.environ.get("STRATA_REF_BIN")
    bin_out = open(bin_path, "wb") if bin_path else None
    off = 0

    def emit(label, vec):
        nonlocal off
        at = -1
        if bin_out is not None:
            at = off
            vec.astype(np.float32).tofile(bin_out)
            off += vec.size
        print(f"{label} abs={np.abs(vec).sum():12.6f} first={vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f} off={at}")

    row = dequant_q8_0({x.name: x for x in g.tensors}["token_embd.weight"], g, tok, N_EMBD)
    r = rms(row) * V("token_embd_norm.weight")
    emit("blk  0 l_out", r)

    blocks = []
    cur = r
    for il in range(N_PRELUDE):
        cur = run_layer(g, il, cur, blocks, W, V)
        emit(f"blk {il:2d} l_out", cur)
    if bin_out is not None:
        bin_out.close()


if __name__ == "__main__":
    main()
