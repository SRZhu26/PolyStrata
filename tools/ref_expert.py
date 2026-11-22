#!/usr/bin/env python3
"""One expert's contribution to blk.0's FFN, in FP64, under the readings a wrong engine would give.

Walks ref_blk0.py's path up to the normed FFN input, then computes that expert's down projection
alone so that its magnitude can be compared with the engine's whole FFN.

    python tools/ref_expert.py FILE.gguf TOKEN EXPERT WEIGHT
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gguf_reader import GGUFFile
from ref_blk0 import (D_REL, EPS, HEAD_DIM, K, N_EMBD, N_FF, N_HEAD, N_KV, QK_CLIP, REL_CLIP, SITU_BETA,
                      SITU_LB, clip, conv, dequant_nvfp4, dequant_q8_0, expert_scale, load, mix, rms, situ)


def main():
    path, tok, exp, wgt = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    g = GGUFFile(path)
    W = lambda n: load(g, n)
    V = lambda n: load(g, n).ravel()
    tens = {x.name: x for x in g.tensors}
    p = "blk.0."

    r = rms(dequant_q8_0(tens["token_embd.weight"], g, tok, N_EMBD)) * V("token_embd_norm.weight")

    xin = rms(r) * V(p + "attn_norm.weight")
    zero_k = np.zeros((K - 1, N_KV * HEAD_DIM))
    k = clip(conv((W(p + "attn_k.weight") @ xin).reshape(1, -1), zero_k, W(p + "attn_k_conv.weight"))[0],
              None, HEAD_DIM, N_KV, QK_CLIP)
    v = conv((W(p + "attn_v.weight") @ xin).reshape(1, -1), zero_k, W(p + "attn_v_conv.weight"))[0]
    vn = np.concatenate([rms(v[h * HEAD_DIM:(h + 1) * HEAD_DIM]) for h in range(N_KV)])
    q = clip(W(p + "attn_q.weight") @ xin, W(p + "attn_q_clip.weight"), HEAD_DIM, N_HEAD, QK_CLIP)
    rr = clip(W(p + "attn_rel.weight") @ xin, W(p + "attn_rel_clip.weight"), D_REL, N_HEAD, REL_CLIP)
    sinks, bias = W(p + "attn_sink_proj.weight") @ xin, W(p + "attn_rel_bias.weight")
    att = np.empty(N_HEAD * HEAD_DIM)
    for h in range(N_HEAD):
        hkv = h // (N_HEAD // N_KV)
        logit = float(q[h * HEAD_DIM:(h + 1) * HEAD_DIM] @ k[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]) / np.sqrt(HEAD_DIM)
        logit += float(bias[0] @ rr[h * D_REL:(h + 1) * D_REL]) - sinks[h]
        best = max(logit, 0.0)
        att[h * HEAD_DIM:(h + 1) * HEAD_DIM] = \
            np.exp(logit - best) / (np.exp(logit - best) + np.exp(-best)) * vn[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]
    att *= 1.0 / (1.0 + np.exp(-(W(p + "attn_gate.weight") @ xin)))
    ao = conv((W(p + "attn_output.weight") @ att).reshape(1, -1), np.zeros((K - 1, N_EMBD)),
              W(p + "attn_post_conv.weight"))[0]
    xn = rms(mix([ao, r], V(p + "ffn_res_query.weight"))) * V(p + "ffn_norm.weight")

    sgu = expert_scale(g, p + "ffn_gate_up_exps", exp)
    sdn = expert_scale(g, p + "ffn_down_exps", exp)
    wgu = dequant_nvfp4(g, tens[p + "ffn_gate_up_exps.weight"], exp, N_FF * 2, N_EMBD) * sgu
    wdn = dequant_nvfp4(g, tens[p + "ffn_down_exps.weight"], exp, N_EMBD, N_FF) * sdn
    print(f"expert {exp} w={wgt} s={sgu:.6e}/{sdn:.6e}")

    def show(tag, dn):
        print(f"  {tag:22s} |dn|={np.abs(dn).sum():11.4f}  w*|dn|={abs(wgt) * np.abs(dn).sum():11.4f}")

    g_, u_ = wgu[:N_FF] @ (wgt * xn), wgu[N_FF:] @ (wgt * xn)
    gn, un = wgu[:N_FF] @ xn, wgu[N_FF:] @ xn
    show("correct", wdn @ situ(g_, u_))
    show("halves swapped", wdn @ situ(u_, g_))
    show("silu", wdn @ (g_ / (1.0 + np.exp(-g_)) * u_))
    show("scale after situ", wdn @ (situ(gn, un) * sgu))
    show("no input weight", wdn @ situ(gn, un))
    show("no output weight", wgt * (wdn @ situ(g_, u_)))
    show("no weights", wdn @ situ(gn, un))


if __name__ == "__main__":
    main()
