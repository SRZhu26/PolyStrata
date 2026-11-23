#!/usr/bin/env python3
"""The prelude's last layer, stage by stage, against the engine's dump of the same names.

`tools/dump_cos.py` only compares `l_out`, so a layer that is 0.9989 has no visible cause.  This runs
`tools/ref_core.run_layer`'s arithmetic for one layer and emits every intermediate under the name the engine's
`dbg()` gives it, so the first stage that parts company is the one to look at.

    STRATA_DUMP_BIN=eng.f32 build-poly/strata-poly.exe ... --tokens 1000 --dump-layers 2> eng.txt
    python tools/diag_stage.py FILE.gguf 1000 5 eng.txt eng.f32
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gguf_reader import GGUFFile
from ref_blk0 import (EPS, HEAD_DIM, K, N_EMBD, N_FF, N_HEAD, N_KV, QK_CLIP, REL_CLIP, SITU_BETA, SITU_LB,
                      W_SCALE, N_EXPERT, N_USED, clip, conv, dequant_q8_0, expert_scale, load, rms, situ)
from ref_core import CORE_END, CORE_START, N_CTRL, N_LAYER, N_PRELUDE_BLOCKS, ink_control
from ref_prelude import BLOCK, D_REL, GLOBAL_LAYERS, N_PRELUDE, conv_dyn, mix_q


def stages(g, il, cur, bank, prev, W, V, pos, out):
    """One layer's intermediates, keyed by the name `Inkling2::layer()` dumps them under."""
    p = f"blk.{il}."
    swa = il not in GLOBAL_LAYERS
    ip = il % BLOCK
    n_q = 1 if swa else 3

    inpSA = cur
    msg = None
    # the pass message: every core layer asks the bank its own, as the C++ reference's per-`ip` loop does
    if CORE_START <= il < CORE_END and prev is not None:
        if il == CORE_START:
            # the bank holds BLOCKS, not layers: the prelude is one block, so cut to it here
            del bank[N_PRELUDE_BLOCKS:]
        prev_bank, prev_out = prev
        states = list(reversed(prev_bank)) + [prev_out]
        q = V(p + "loop_msg_query.weight").reshape(N_CTRL, N_EMBD)
        msg = mix_q(states, q)
        out["msg"] = np.concatenate(msg)

    inputs = [cur.copy(), cur.copy(), cur.copy()]
    if il > 0:
        q = V(p + "attn_res_query.weight").reshape(n_q, N_EMBD)
        m = mix_q(bank + [cur], q)
        inputs = [m[0].copy() for _ in range(3)] if n_q == 1 else [m[j].copy() for j in range(3)]

    # the control pulls each of the mixes toward the pass before's message, before the input norms
    if msg is not None:
        for j in range(3):
            down = W(p + f"loop_{['q', 'k', 'v'][j]}_ctrl_down.weight")
            gain = W(p + f"loop_{['q', 'k', 'v'][j]}_ctrl_gain.weight")
            forget = W(p + f"loop_{['q', 'k', 'v'][j]}_ctrl_forget.weight")
            inputs[j] = ink_control(inputs[j], msg[j], down, gain, forget, gain.shape[0])

    # a window layer normed once shares the one input with all three, as the reference does - but a pass that
    # has messages has already parted them, so each is normed on its own (`is_swa && messages` in the C++)
    if swa and msg is not None:
        inputs = [rms(inputs[j]) * V(p + "attn_norm.weight") for j in range(3)]
    elif swa:
        n = rms(inputs[0]) * V(p + "attn_norm.weight")
        inputs = [n.copy(), n.copy(), n.copy()]
    else:
        inputs = [rms(inputs[0]) * V(p + "attn_q_inp_norm.weight"),
                  rms(inputs[1]) * V(p + "attn_k_inp_norm.weight"),
                  rms(inputs[2]) * V(p + "attn_v_inp_norm.weight")]
    out["attn_in"] = inputs[0].copy()
    out["attn_in1"] = inputs[1].copy()
    out["attn_in2"] = inputs[2].copy()

    Qraw = W(p + "attn_q.weight") @ inputs[0]
    Kraw = W(p + "attn_k.weight") @ inputs[1]
    Vraw = W(p + "attn_v.weight") @ inputs[2]
    Rraw = W(p + "attn_rel.weight") @ inputs[0]

    zero = np.zeros((K - 1, N_KV * HEAD_DIM))
    kdyn = None if swa else (W(p + "attn_k_conv_proj.weight") @ inputs[1]).reshape(N_KV, K)
    vdyn = None if swa else (W(p + "attn_v_conv_proj.weight") @ inputs[2]).reshape(N_KV, K)
    k = conv_dyn(Kraw.reshape(1, -1), zero, W(p + "attn_k_conv.weight"), kdyn)[0]
    v = conv_dyn(Vraw.reshape(1, -1), zero, W(p + "attn_v_conv.weight"), vdyn)[0]

    q = clip(Qraw, W(p + "attn_q_clip.weight"), HEAD_DIM, N_HEAD, QK_CLIP)
    out["kraw"] = Kraw.copy()
    out["vraw"] = Vraw.copy()
    k = clip(k, None, HEAD_DIM, N_KV, QK_CLIP)
    vn = np.concatenate([rms(v[h * HEAD_DIM:(h + 1) * HEAD_DIM]) for h in range(N_KV)])
    r = clip(Rraw, W(p + "attn_rel_clip.weight"), D_REL, N_HEAD, REL_CLIP)
    tau = 1.0 + 0.1 * np.log(max(1.0, (pos + 1) / 8192.0))
    if not swa:
        q = q * tau
        r = r * tau
    out["q_clip"] = q.copy()
    out["k_conv"] = k.copy()
    out["vn"] = vn.copy()
    out["r_clip"] = r.copy()

    sinks = W(p + "attn_sink_proj.weight") @ inputs[0]
    bias = W(p + "attn_rel_bias.weight")
    kqb = np.array([bias[0] @ r[h * D_REL:(h + 1) * D_REL] for h in range(N_HEAD)]) - sinks
    out["sinks"] = sinks.copy()
    out["kqb"] = kqb.copy()

    att = np.empty(N_HEAD * HEAD_DIM)
    for h in range(N_HEAD):
        hkv = h // (N_HEAD // N_KV)
        kh = k[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]
        logit = float(q[h * HEAD_DIM:(h + 1) * HEAD_DIM] @ kh) / np.sqrt(HEAD_DIM) + kqb[h]
        best = max(logit, 0.0)
        w0 = np.exp(logit - best) / (np.exp(logit - best) + np.exp(-best))
        att[h * HEAD_DIM:(h + 1) * HEAD_DIM] = w0 * vn[hkv * HEAD_DIM:(hkv + 1) * HEAD_DIM]
    out["att_raw"] = att.copy()

    gate = 1.0 / (1.0 + np.exp(-(W(p + "attn_gate.weight") @ inputs[0])))
    out["gate"] = (W(p + "attn_gate.weight") @ inputs[0])   # the engine dumps the logit, before the sigmoid
    att = att * gate
    out["att_gated"] = att.copy()
    ao = conv_dyn((W(p + "attn_output.weight") @ att).reshape(1, -1), np.zeros((K - 1, N_EMBD)),
                  W(p + "attn_post_conv.weight"), None)[0]
    out["ao"] = ao.copy()

    if ip == 0:
        bank.append(inpSA)
        cur = ao.copy()
    else:
        cur = inpSA + ao
    out["ffn_inp"] = cur.copy()

    m = mix_q(bank + [cur], V(p + "ffn_res_query.weight").reshape(1, N_EMBD))
    xn = m[0].copy()
    if msg is not None:
        xn = ink_control(xn, msg[3], W(p + "loop_ffn_ctrl_down.weight"),
                         W(p + "loop_ffn_ctrl_gain.weight"), W(p + "loop_ffn_ctrl_forget.weight"),
                         W(p + "loop_ffn_ctrl_gain.weight").shape[0])
    xn = rms(xn) * V(p + "ffn_norm.weight")
    out["ffn_in"] = xn.copy()   # the engine dumps the normed FFN input as `ffn_in`
    logits = W(p + "ffn_gate_inp.weight") @ xn
    probs = 1.0 / (1.0 + np.exp(-logits))
    sel = np.argsort(-(probs + V(p + "exp_probs_b.bias")))[:N_USED]
    w = probs[sel] / probs[sel].sum() * W_SCALE
    tens = {x.name: x for x in g.tensors}
    acc = np.zeros(N_EMBD)
    for kk, e in enumerate(sel):
        from ref_core import expert_matrix
        wgu = expert_matrix(g, tens, p + "ffn_gate_up_exps", int(e), N_FF * 2, N_EMBD)
        wdn = expert_matrix(g, tens, p + "ffn_down_exps", int(e), N_EMBD, N_FF)
        acc += w[kk] * (wdn @ situ(wgu[:N_FF] @ xn, wgu[N_FF:] @ xn))
    out["ffn"] = acc.copy()
    fn = conv_dyn(acc.reshape(1, -1), np.zeros((K - 1, N_EMBD)), W(p + "ffn_post_conv.weight"), None)[0]
    out["ffn"] = fn.copy()
    out["l_out"] = cur + fn
    return cur + fn


def main():
    path, tok, il, eng_txt, eng_bin = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    want_pass = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    g = GGUFFile(path)
    W = lambda n: load(g, n)
    V = lambda n: load(g, n).ravel()

    row = dequant_q8_0({x.name: x for x in g.tensors}["token_embd.weight"], g, tok, N_EMBD)
    cur = rms(row) * V("token_embd_norm.weight")
    bank, prev = [], None
    prelude = None
    got = None
    for p in range(want_pass + 1):
        if p == 0:
            layers = list(range(N_PRELUDE)) + list(range(CORE_START, CORE_END))
        else:
            # every pass after the first starts from the prelude's output, not from the pass before's
            layers = list(range(CORE_START, CORE_END))
            cur = prelude
        for i in layers:
            stages_out = {}
            cur = stages(g, i, cur, bank, prev, W, V, 0, stages_out)
            if i == il and p == want_pass:
                got = stages_out
            if p == 0 and i == N_PRELUDE - 1:
                prelude = cur.copy()
            # the previous pass's blocks AFTER the first core block, and its output
            if p < want_pass and i == CORE_END - 1:
                prev = (bank[N_PRELUDE_BLOCKS + 1:], cur)

    # the engine dumps the same name once a pass, so the Nth occurrence is this pass's
    import re
    raw = np.memmap(eng_bin, dtype=np.float32, mode="r")
    seen = {}
    eng = {}
    for line in open(eng_txt, errors="ignore"):
        m = re.match(r"dump pos=0 blk=(-?\d+) (\S+)\s+n=(\d+) sum=\S+ abs=\S+ first=\S+ \S+ \S+ off=(-?\d+)", line)
        if not m:
            continue
        b, name, n, off = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if b != il:
            continue
        k = seen.get(name, 0)
        seen[name] = k + 1
        # `msg` exists only once a pass has messages, so its occurrences start at pass 1
        if k == (want_pass - 1 if name == "msg" else want_pass):
            eng[name] = np.asarray(raw[off:off + n], dtype=np.float64)

    print(f"pass {want_pass}, blk {il}:")
    print(f"{'stage':>10} {'n':>6} {'cosine':>12} {'max|d|':>10} {'ref abs':>10}")
    for name, vec in got.items():
        e = eng.get(name)
        if e is None:
            print(f"{name:>10} {'--':>6} missing from the engine")
            continue
        if len(e) != len(vec):
            # `kqb` is the whole bias table per head, the reference only the distance-0 entry
            if len(e) % len(vec) == 0:
                e = e[::len(e) // len(vec)][:len(vec)]
            else:
                print(f"{name:>10} engine {len(e)}, reference {len(vec)}")
                continue
        c = float(e @ vec / (np.linalg.norm(e) * np.linalg.norm(vec)))
        print(f"{name:>10} {len(vec):6d} {c:12.9f} {np.abs(e - vec).max():10.4f} {np.abs(vec).sum():10.2f}")


if __name__ == "__main__":
    main()
