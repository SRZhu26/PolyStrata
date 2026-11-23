#!/usr/bin/env python3
"""An FP64 reference for the looped core of Inkling 2 Mini, at position 0, one token.

`ref_prelude.py` stops at the prelude (blk.0 .. blk.5).  This runs the whole model: the prelude, the core
[6, 18) three times, the epilogue [18, 24), and the output.  It exists for the plan's Phase 1 gate, whose
points past blk.5 - blk.6 on passes 1 to 3, blk.23 and the final logits - are all in the core.

The core's depth attention is what makes it more than a repeat.  At the start of a pass after the first, the
bank is cut back to the prelude's blocks and rebuilt from `pass_messages()`: the pass before's core blocks
*after the first* (`n_prev_bank_ = n_bank_ - (n_prelude_blocks_ + 1)`, i.e. blk.12's input, not blk.6's) and
that pass's output, newest first.  Each of the four control taps then pulls the input it mixes toward the
message: x = x - tanh(forget . gate) * x + tanh(gain . gate) * msg, with gate = down @ concat(rms(x), rms(msg)).

    python tools/ref_core.py FILE.gguf TOKEN

Every stage's `l_out` is printed beside the engine's `--dump-layers` line of the same name, and the last
token's logits after `finish_logits`.
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
from ref_prelude import BLOCK, D_REL, GLOBAL_LAYERS, N_PRELUDE, conv_dyn, mix_q

CORE_START = 6
CORE_END = 18
LOOP_COUNT = 3
N_LAYER = 24
N_PRELUDE_BLOCKS = CORE_START // BLOCK      # 1: a block is six layers, the prelude is one block
N_CTRL = 4                                    # three for Q, K, V and one for the FFN
N_VOCAB = None                                # filled from the file


def ink_control(x, msg, down, gain, forget, rank):
    """The looped core's control: (1 - tanh(forget.gate)) * x + tanh(gain.gate) * msg.

    `down` is {rank, 2*n_embd} and read row-major; `gain` and `forget` are {rank, n_embd} and the value of
    (rank r, embd e) sits at e*rank + r, not r*n_embd + e (inkling2_kernels.cu:362).
    """
    gate = down @ np.concatenate([rms(x), rms(msg)])
    g = gain @ gate
    f = forget @ gate
    return x - np.tanh(f) * x + np.tanh(g) * msg


def run_layer(g, il, cur, bank, prev, W, V, pos=0):
    """One layer from its input to `l_out`, with `prev` = (bank, out) of the pass before or None.

    `bank` is the depth attention's bank as `Inkling2::push_bank()` builds it: a block's first layer
    appends its input, and the core's first layer of a pass cuts it back to the prelude's blocks.
    """
    p = f"blk.{il}."
    swa = il not in GLOBAL_LAYERS
    ip = il % BLOCK
    n_q = 1 if swa else 3
    in_core = CORE_START <= il < CORE_END

    inpSA = cur
    # pass_messages(): the bank is cut to the prelude's blocks at the pass's first core layer, and
    # every core layer then asks the bank its own message (its own `loop_msg_query`), as the C++
    # reference's `build_layer(..., mix(bank, bank_out, loop.msg_query), &loop)` does per `ip`
    msg = None
    if in_core and prev is not None:
        if il == CORE_START:
            del bank[N_PRELUDE_BLOCKS:]
        prev_bank, prev_out = prev
        states = list(reversed(prev_bank)) + [prev_out]
        q = V(p + "loop_msg_query.weight").reshape(N_CTRL, N_EMBD)
        msg = mix_q(states, q)

    inputs = [cur.copy(), cur.copy(), cur.copy()]
    if il > 0:                                          # a block's first layer has no query
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

    # A window layer normed once shares the one input with all three - but a pass with messages has already
    # parted them, and the reference norms each of the three on its own there (`is_swa && messages`)
    if swa and msg is not None:
        inputs = [rms(inputs[j]) * V(p + "attn_norm.weight") for j in range(3)]
    elif swa:
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
    kdyn = None if swa else (W(p + "attn_k_conv_proj.weight") @ inputs[1]).reshape(N_KV, K)
    vdyn = None if swa else (W(p + "attn_v_conv_proj.weight") @ inputs[2]).reshape(N_KV, K)
    k = conv_dyn(Kraw.reshape(1, -1), zero, W(p + "attn_k_conv.weight"), kdyn)[0]
    v = conv_dyn(Vraw.reshape(1, -1), zero, W(p + "attn_v_conv.weight"), vdyn)[0]

    q = clip(Qraw, W(p + "attn_q_clip.weight"), HEAD_DIM, N_HEAD, QK_CLIP)
    k = clip(k, None, HEAD_DIM, N_KV, QK_CLIP)
    vn = np.concatenate([rms(v[h * HEAD_DIM:(h + 1) * HEAD_DIM]) for h in range(N_KV)])
    r = clip(Rraw, W(p + "attn_rel_clip.weight"), D_REL, N_HEAD, REL_CLIP)
    tau = 1.0 + 0.1 * np.log(max(1.0, (pos + 1) / 8192.0))
    if not swa:
        q = q * tau
        r = r * tau

    sinks = W(p + "attn_sink_proj.weight") @ inputs[0]
    bias = W(p + "attn_rel_bias.weight")
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

    if ip == 0:
        bank.append(inpSA)
        cur = ao.copy()
    else:
        cur = inpSA + ao
    ffn_inp = cur.copy()

    m = mix_q(bank + [cur], V(p + "ffn_res_query.weight").reshape(1, N_EMBD))
    # the FFN's own control tap, the fourth: it reads the mix before the norm, as the C++ reference's
    # graph does (control(cur, ...) then build_norm(cur, ffn_norm)), and the norm follows it
    xn = m[0].copy()
    if msg is not None:
        xn = ink_control(xn, msg[3], W(p + "loop_ffn_ctrl_down.weight"),
                         W(p + "loop_ffn_ctrl_gain.weight"), W(p + "loop_ffn_ctrl_forget.weight"),
                         W(p + "loop_ffn_ctrl_gain.weight").shape[0])
    xn = rms(xn) * V(p + "ffn_norm.weight")

    logits = W(p + "ffn_gate_inp.weight") @ xn
    probs = 1.0 / (1.0 + np.exp(-logits))
    sel = np.argsort(-(probs + V(p + "exp_probs_b.bias")))[:N_USED]
    w = probs[sel] / probs[sel].sum() * W_SCALE
    tens = {x.name: x for x in g.tensors}
    acc = np.zeros(N_EMBD)
    for kk, e in enumerate(sel):
        wgu = expert_matrix(g, tens, p + "ffn_gate_up_exps", int(e), N_FF * 2, N_EMBD)
        wdn = expert_matrix(g, tens, p + "ffn_down_exps", int(e), N_EMBD, N_FF)
        acc += w[kk] * (wdn @ situ(wgu[:N_FF] @ xn, wgu[N_FF:] @ xn))
    fn = conv_dyn(acc.reshape(1, -1), np.zeros((K - 1, N_EMBD)), W(p + "ffn_post_conv.weight"), None)[0]
    return ffn_inp + fn


def expert_matrix(g, tens, name, expert, nrow, ncol):
    """An expert's matrix as fp64: NVFP4 times its global scale, or Q8_0 as its blocks decode it.

    The epilogue's experts (blk.20 .. blk.23) are Q8_0, which carries its scale in every block and has no
    `.scale` tensor beside it, so expert_scale() would not find one.  The tensor's rows are the experts stacked
    one after another (a row is ncol values, ne0), so expert e's own rows start at e * nrow.
    """
    t = tens[name + ".weight"]
    if t.type_name == "Q8_0":
        return np.stack([dequant_q8_0(t, g, expert * nrow + r, ncol) for r in range(nrow)])
    return dequant_nvfp4(g, t, expert, nrow, ncol) * expert_scale(g, name, expert)


def main():
    global N_VOCAB
    path = pathlib.Path(sys.argv[1])
    tok = int(sys.argv[2])
    g = GGUFFile(path)
    W = lambda n: load(g, n)
    V = lambda n: load(g, n).ravel()
    N_VOCAB = len(g.metadata["tokenizer.ggml.tokens"])

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

    bank, prev = [], None
    prelude = None
    for p in range(LOOP_COUNT):
        # the prelude runs once, the core three times, and each pass's last core layer hands the next its
        # blocks after the first and its output
        if p == 0:
            layers = list(range(N_PRELUDE)) + list(range(CORE_START, CORE_END))
        else:
            # every pass after the first starts from the prelude's output, not from the pass before's:
            # docs/models/reference/inkling2-loop.cpp sets `cur = prelude` and `blocks = prelude_blocks`
            layers = list(range(CORE_START, CORE_END))
            r = prelude
        for il in layers:
            r = run_layer(g, il, r, bank, prev, W, V, pos=0)
            emit(f"blk {il:2d} l_out", r)
            if p == 0 and il == N_PRELUDE - 1:
                prelude = r.copy()
            if p < LOOP_COUNT - 1 and il == CORE_END - 1:
                prev = (bank[N_PRELUDE_BLOCKS + 1:], r)
    for il in range(CORE_END, N_LAYER):
        r = run_layer(g, il, r, bank, prev, W, V, pos=0)
        emit(f"blk {il:2d} l_out", r)

    # the output: the last mix of the bank, then the output norm and the matrix.  The matrix has
    # 966 pad rows past the vocabulary (output.weight is [2304, 201024], the tokenizer 200,058), and
    # the engine's softcapped logits stop at the vocabulary, so the reference drops the pad too.
    states = bank + [r]
    m = mix_q(states, V("output_res_query.weight").reshape(1, N_EMBD))
    h = rms(m[0]) * V("output_norm.weight")
    logits = (W("output.weight") @ h)[:N_VOCAB]
    emit("logits", logits)
    if bin_out is not None:
        bin_out.close()


if __name__ == "__main__":
    main()
