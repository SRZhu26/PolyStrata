"""128k head-to-head on this laptop: llama.cpp against strata-poly, one prompt, one token count.

Both engines are driven through their OpenAI-compatible HTTP servers with the SAME prompt text, the
same sampler settings and a 131,072-token context, so the figures are a measurement of the two engines
and not of two different prompts.  The prompt file is built by tools/make_long_prompt.py and checked with
llama-tokenize before the run: a prompt that the two tokenizers count differently would make the prefill
figures incomparable.

    python tools/bench_128k_local.py --out DIR --models name=PATH[,name=PATH] --engines llama,strata

Per run: start, warm-up (discarded), the 128k prompt, 256 generated tokens, stop.  Peak VRAM comes
from nvidia-smi polling, RAM from the engine's process tree.  A run that dies is recorded as failed, not
as a zero, so a model that does not fit at 128k cannot look slow instead of impossible.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
LLAMA = Path("C:/Users/zshua/models/llama-b11669-bin-win-cuda-13.4-x64")
CUDA_BIN = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin")
PORT = 8191


def http(url, body=None, timeout=5400):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vram_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return int(out[0])
    except Exception:
        return -1


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop = pid, False
        self.vram = self.rss = self.sysused = self.swap = 0

    def run(self):
        while not self.stop:
            self.vram = max(self.vram, vram_used_mib())
            try:
                p = psutil.Process(self.pid)
                self.rss = max(self.rss, sum(c.memory_info().rss for c in [p] + p.children(recursive=True)))
            except psutil.Error:
                pass
            m = psutil.virtual_memory()
            self.sysused = max(self.sysused, m.used)
            self.swap = max(self.swap, psutil.swap_memory().used)
            time.sleep(1.0)


def strata_config(model, out, tok_dir, ctx, kv, threads, extra):
    cfg = {
        "exe": str(ROOT / "build-poly" / "strata-poly.exe"),
        "args": ["--model", str(model), "--max-context", str(ctx), "--kv", kv, "--threads", str(threads)],
        "cwd": str(ROOT),
        "log": str(Path(out) / ("engine-%s.log" % Path(model).stem)),
        "tokenizer": str(tok_dir),
        "model_name": Path(model).stem.lower(),
        "port": PORT,
    }
    cfg["args"] += extra
    p = Path(out) / ("strata-cfg-%s.json" % Path(model).stem)
    p.write_text(json.dumps(cfg, indent=1))
    return p


def start(cmd, out, tag, env=None):
    log = open(Path(out) / ("server-%s-%d.log" % (tag, int(time.time()))), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    t0 = time.time()
    while time.time() - t0 < 1800:
        if proc.poll() is not None:
            log.close()
            return None, None, None, log.name
        try:
            if http("http://127.0.0.1:%d/health" % PORT, timeout=5).get("status", "ok") == "ok":
                return proc, log, time.time() - t0, log.name
        except Exception:
            pass
        time.sleep(2)
    proc.kill()
    log.close()
    return None, None, None, log.name


def stop(proc, log):
    if not proc:
        return
    try:
        p = psutil.Process(proc.pid)
        kids = p.children(recursive=True)
        for c in kids + [p]:
            try:
                c.terminate()
            except psutil.Error:
                pass
        psutil.wait_procs(kids + [p], timeout=30)
        for c in kids + [p]:
            try:
                if c.is_running():
                    c.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    try:
        log.close()
    except Exception:
        pass
    time.sleep(4)


def chat(prompt, max_tokens, temp=0.0, seed=1):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temp, "seed": seed, "stream": False, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if temp > 0.0:
        # a greedy decode is a fair speed number but a poor quality one: a model in a slightly wrong state
        # cycles, where with a nucleus of the same distribution it does not.  The quality runs ask for this.
        body["top_p"] = 0.95
        body["top_k"] = 40
    t = time.time()
    r = http("http://127.0.0.1:%d/v1/chat/completions" % PORT, body)
    wall = time.time() - t
    tm, u = r.get("timings") or {}, r.get("usage") or {}
    return {"wall_s": round(wall, 2),
            "prompt_n": tm.get("prompt_n", u.get("prompt_tokens")),
            "prompt_ms": tm.get("prompt_ms"),
            "prompt_tps": tm.get("prompt_per_second"),
            "pred_n": tm.get("predicted_n", u.get("completion_tokens")),
            "pred_ms": tm.get("predicted_ms"),
            "pred_tps": tm.get("predicted_per_second"),
            "text": (r["choices"][0]["message"].get("content") or "")[:1500]}


def run_one(engine, model, a, prompt, out):
    tag = "%s-%s-%dk" % (engine, Path(model).stem[:24], a.context // 1024)
    if engine == "llama":
        # llama.cpp spells the same cache type f16, where strata spells it fp16
        kv = {"fp16": "f16", "q8_0": "q8_0", "f32": "f32", "bf16": "bf16"}.get(a.kv, a.kv)
        cmd = [str(Path(a.llama_dir) / "llama-server.exe"), "-m", str(model), "--port", str(PORT), "--host", "127.0.0.1",
               "-np", "1", "--no-webui", "--jinja", "-c", str(a.context),
               "-ctk", kv, "-ctv", kv, "-fa", "on", "--fit-target", "1024"] + a.llama_args
    else:
        cfg = strata_config(model, out, a.tokenizer, a.context, a.kv, a.threads, a.strata_args)
        cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
               "--config", str(cfg), "--port", str(PORT)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
                PATH=str(CUDA_BIN) + os.pathsep + os.environ.get("PATH", ""))
    print("[%s] starting ..." % tag, flush=True)
    proc, log, load_s, logname = start(cmd, out, tag, env)
    if not proc:
        print("[%s] DID NOT START (%s)" % (tag, logname), flush=True)
        return {"engine": engine, "model": Path(model).name, "status": "did_not_start", "log": logname}
    samp = Sampler(proc.pid)
    samp.start()
    rec = {"engine": engine, "model": Path(model).name, "status": "ok", "load_s": round(load_s, 1)}
    try:
        chat("Say hello.", 8)                                  # warm-up, discarded
        r = chat(prompt, a.gen, temp=a.temp)
        rec.update(r)
        rec["vram_peak_mib"] = samp.vram
        rec["rss_peak_gib"] = round(samp.rss / 2**30, 2)
        rec["sys_used_peak_gib"] = round(samp.sysused / 2**30, 2)
        rec["swap_peak_gib"] = round(samp.swap / 2**30, 2)
        print("[%s] prompt %s tok at %s tok/s, %s generated at %s tok/s, TTFT %s s, VRAM %s MiB, RSS %s GiB"
              % (tag, r.get("prompt_n"), r.get("prompt_tps"), r.get("pred_n"), r.get("pred_tps"),
                 None if r.get("prompt_ms") is None else round(r["prompt_ms"] / 1000, 1),
                 samp.vram, round(samp.rss / 2**30, 2)), flush=True)
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = repr(e)[:300]
        print("[%s] FAILED %r" % (tag, e), flush=True)
    finally:
        samp.stop = True
        stop(proc, log)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True, help="name=PATH,name=PATH")
    ap.add_argument("--engines", default="strata")
    ap.add_argument("--llama-dir", default=str(LLAMA),
                    help="directory containing llama-server.exe (needs inkling2_loop support: PR #32704 or Unsloth b12588+)")
    ap.add_argument("--prompt", default=str(ROOT / "bench" / "prompt-128k.txt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "bench" / "tok-inkling2" / "tokenizer"))
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature; 0 is the engine's greedy default")
    ap.add_argument("--kv", default="q8_0")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--llama-args", nargs="*", default=[])
    ap.add_argument("--strata-args", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    prompt = Path(a.prompt).read_text(encoding="utf-8")
    models = [m.split("=", 1) for m in a.models.split(",")]
    recs = []
    for engine in a.engines.split(","):
        for name, path in models:
            rec = run_one(engine, path, a, prompt, out)
            rec["name"] = name
            recs.append(rec)
            with open(out / "raw" / "runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    (out / "runs.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    print("\n| model | engine | status | load s | prompt tok/s | decode tok/s | TTFT s | VRAM MiB | RSS GiB | swap GiB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["name"], r["engine"], r["status"], r.get("load_s", "-"),
            None if r.get("prompt_tps") is None else round(r["prompt_tps"], 1),
            None if r.get("pred_tps") is None else round(r["pred_tps"], 2),
            None if r.get("prompt_ms") is None else round(r["prompt_ms"] / 1000, 1),
            r.get("vram_peak_mib", "-"), r.get("rss_peak_gib", "-"), r.get("swap_peak_gib", "-")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
