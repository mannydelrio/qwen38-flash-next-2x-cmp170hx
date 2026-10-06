"""Teacher-forced quality check (Strata's method): run fixed texts through the server with prompt_logprobs and
compare configs position by position: same top-1 as the reference, approx KL over the top-k, and perplexity.
  python3 tf_compare.py dump OUT.json        # against the running server
  python3 tf_compare.py compare REF.json NEW.json
Texts are built from files on TinyBeast so every config sees identical input."""
import json, os, math, sys, time, uuid, urllib.request
from pathlib import Path

KEY = os.environ.get("VLLM_API_KEY", "")
URL = "http://127.0.0.1:18084/v1/completions"
P = Path(os.environ.get("FLASHNEXT_ROOT", "."))
TOPK = 10


def texts():
    code = (P / "vllm-backport/vllm/v1/core/sched/scheduler.py").read_text()[2000:8000]
    doc = Path("/usr/share/common-licenses/GPL-3").read_text()[3000:9000]
    chat = (P / "ple-ssd/HANDOFF.md").read_text()[:5000]
    return {"code": code, "document": doc, "notes": chat}


def dump(out):
    res = {}
    for name, text in texts().items():
        body = {"model": "tinybeast-flashnext-uncensored", "prompt": text, "max_tokens": 1, "temperature": 0,
                "prompt_logprobs": TOPK, "cache_salt": str(uuid.uuid4())}
        req = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        t = time.time()
        d = json.loads(urllib.request.urlopen(req, timeout=600).read())
        plp = d["choices"][0].get("prompt_logprobs") or d.get("prompt_logprobs")
        rows = []
        for pos in plp[1:]:  # position 0 has no prediction
            top = sorted(((int(k), v["logprob"], v.get("rank", 99)) for k, v in pos.items()), key=lambda x: x[2])
            rows.append({str(t_id): lp for t_id, lp, _ in top})
        # the actual token at each position (for perplexity) comes from the server's own tokenizer
        treq = urllib.request.Request(URL.replace("/v1/completions", "/tokenize"),
                                      json.dumps({"model": body["model"], "prompt": text}).encode(),
                                      {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        ids = json.loads(urllib.request.urlopen(treq, timeout=60).read())["tokens"]
        assert len(ids) == len(plp), (len(ids), len(plp))
        res[name] = {"rows": rows, "ids": ids[1:]}
        print(f"  {name}: {len(rows)} positions in {time.time() - t:.1f}s", flush=True)
    json.dump(res, open(out, "w"))


def compare(ref_path, new_path):
    ref, new = json.load(open(ref_path)), json.load(open(new_path))
    tot = {"n": 0, "same": 0, "kl": 0.0, "nll_r": 0.0, "nll_n": 0.0, "nll_cnt": 0}
    print(f"{'text':<10}{'positions':>10}{'same top-1':>12}{'mean KL':>10}{'ppl ref':>10}{'ppl new':>10}")
    for name in ref:
        R, N = ref[name]["rows"], new[name]["rows"]
        ids = ref[name]["ids"]
        n = min(len(R), len(N))
        same = kl = nr = nn = 0.0
        cnt = 0
        for i in range(n):
            r, m = R[i], N[i]
            if max(r, key=r.get) == max(m, key=m.get):
                same += 1
            floor = min(m.values())  # tokens outside NEW's top-k get its k-th logprob (upper bound)
            kl += sum(math.exp(lp) * (lp - m.get(t, floor)) for t, lp in r.items())
            if ids and i < len(ids):
                t = str(ids[i])
                if t in r and t in m:
                    nr -= r[t]; nn -= m[t]; cnt += 1
        tot["n"] += n; tot["same"] += same; tot["kl"] += kl
        tot["nll_r"] += nr; tot["nll_n"] += nn; tot["nll_cnt"] += cnt
        pr = math.exp(nr / cnt) if cnt else float("nan"); pn = math.exp(nn / cnt) if cnt else float("nan")
        print(f"{name:<10}{n:>10}{100 * same / n:>11.1f}%{kl / n:>10.4f}{pr:>10.3f}{pn:>10.3f}")
    c = tot["nll_cnt"]
    print(f"{'ALL':<10}{tot['n']:>10}{100 * tot['same'] / tot['n']:>11.1f}%{tot['kl'] / tot['n']:>10.4f}"
          f"{math.exp(tot['nll_r'] / c):>10.3f}{math.exp(tot['nll_n'] / c):>10.3f}"
          f"   (nats/token new-ref: {(tot['nll_n'] - tot['nll_r']) / c:+.4f})")


if __name__ == "__main__":
    {"dump": lambda: dump(sys.argv[2]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
