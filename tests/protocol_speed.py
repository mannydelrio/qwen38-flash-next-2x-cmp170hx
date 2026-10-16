"""Speed test using the protocol other CMP 170HX recipes publish (e.g. Morrowmake's GLM-5.3-Flash recipe), so
numbers compare across setups:
  decode  - three prompts, verbatim from MiaAI-Lab's tests/bench_decode.py (used as data, with attribution:
            github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks): structured / code / prose. Temperature 0,
            thinking off, 400-token cap, streaming.
            1 user: median of N streaming rates, (completion_tokens - 1) / (end - first token).
            2/4/8 users: median of 3 runs of completion tokens / batch wall time (includes prefill and client overhead).
  prefill - real text at 23,945 / 34,299 / 37,905 tokens, unique salt first so the prefix cache cannot hit,
            max_tokens 8, prompt_tokens / time to first token, median of 9.
Usage: python3 tests/protocol_speed.py --out results.json   (API key, if any, from VLLM_API_KEY; prefill text from
TEXT_GLOB, default the vLLM fork's docs under $FLASHNEXT_ROOT)
"""
import argparse, glob, json, os, random, statistics, threading, time, urllib.request, uuid

PROMPTS = {  # verbatim from MiaAI-Lab tests/bench_decode.py
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
    "code": ("Write a Python function named clamp_range that takes a list of ints and "
             "returns a new list with each value clamped to [0, 50]. Include a short docstring."),
    "prose": ("Write a detailed step-by-step explanation of how a hash map works, "
              "including collision handling, resizing, and time complexity. Be thorough."),
}
PREFILL_TARGETS = (23945, 34299, 37905)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:18084")
ap.add_argument("--model", default="tinybeast-flashnext-uncensored")
ap.add_argument("--out", required=True)
ap.add_argument("--single-runs", type=int, default=10)
ap.add_argument("--multi-runs", type=int, default=3)
ap.add_argument("--levels", default="2,4,8")
ap.add_argument("--text-glob", default=os.environ.get("TEXT_GLOB", os.path.join(
    os.environ.get("FLASHNEXT_ROOT", "/opt/flashnext"), "vllm-backport/docs/**/*.md")))
ap.add_argument("--skip-prefill", action="store_true")
ap.add_argument("--skip-decode", action="store_true")
args = ap.parse_args()
KEY = os.environ.get("VLLM_API_KEY", "")
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
results = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "base": args.base, "decode": {}, "prefill": []}


def save():
    json.dump(results, open(args.out, "w"), indent=1)


def post(path, body, timeout=600):
    req = urllib.request.Request(args.base + path, json.dumps(body).encode(), HDR)
    return urllib.request.urlopen(req, timeout=timeout)


def metrics():
    out = {}
    raw = urllib.request.urlopen(urllib.request.Request(args.base + "/metrics", headers=HDR), timeout=10).read().decode()
    for line in raw.splitlines():
        if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prefix_cache_hits_total",
                            "vllm:prefix_cache_queries_total", "vllm:request_success_total")):
            name = line.split("{")[0].split(" ")[0]
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def stream(prompt, max_tokens, out):
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; usage = None; text = []
    with post("/v1/chat/completions", body, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece:
                    if first is None:
                        first = time.time()
                    text.append(piece)
    end = time.time()
    out.update(t0=t0, first=first, end=end, completion_tokens=(usage or {}).get("completion_tokens", 0),
               prompt_tokens=(usage or {}).get("prompt_tokens", 0), text="".join(text))
    return out


def decode_rate(o):
    return (o["completion_tokens"] - 1) / (o["end"] - o["first"]) if o["first"] and o["completion_tokens"] > 1 else None


def busy_note():
    m = metrics()
    return {"running": m.get("vllm:num_requests_running", 0), "waiting": m.get("vllm:num_requests_waiting", 0)}


if not args.skip_decode:
    for name, prompt in PROMPTS.items():
        stream(prompt, 400, {})  # warm-up, as in bench_decode.py
        rates, samples = [], []
        for i in range(args.single_runs):
            pre = busy_note()
            o = stream(prompt, 400, {})
            rates.append(decode_rate(o))
            samples.append({"rate": round(rates[-1], 1), "tokens": o["completion_tokens"], "ttft": round(o["first"] - o["t0"], 3),
                            "busy_before": pre})
        results["decode"].setdefault(name, {})["1"] = {"median": round(statistics.median(rates), 1), "runs": samples,
                                                       "sample_text": o["text"][:300]}
        print(f"{name:10s} 1 user  median {statistics.median(rates):6.1f} tok/s  ({', '.join(f'{r:.0f}' for r in rates)})", flush=True)
        save()
        for level in [int(x) for x in args.levels.split(",")]:
            aggs, runs = [], []
            for rep in range(args.multi_runs):
                pre = busy_note()
                outs = [{} for _ in range(level)]
                t_cell = time.time()
                th = [threading.Thread(target=stream, args=(prompt, 400, outs[i])) for i in range(level)]
                [t.start() for t in th]; [t.join() for t in th]
                wall = max(o["end"] for o in outs) - t_cell
                toks = sum(o["completion_tokens"] for o in outs)
                aggs.append(toks / wall)
                runs.append({"agg": round(aggs[-1], 1), "tokens": toks, "wall": round(wall, 2), "busy_before": pre})
            results["decode"][name][str(level)] = {"median": round(statistics.median(aggs), 1), "runs": runs}
            print(f"{name:10s} {level} users median {statistics.median(aggs):6.1f} tok/s total  ({', '.join(f'{a:.0f}' for a in aggs)})", flush=True)
            save()

if not args.skip_prefill:
    files = sorted(glob.glob(args.text_glob, recursive=True))
    random.Random(7).shuffle(files)
    corpus = "\n\n".join(open(f, errors="replace").read() for f in files)
    ids = json.load(post("/tokenize", {"model": args.model, "prompt": corpus[:700_000], "add_special_tokens": False}))["tokens"]
    rates = []
    for idx, target in enumerate(PREFILL_TARGETS):
        start = idx * 40000  # a different stretch of the corpus for each length
        for rep in range(3):
            salt = f"[{uuid.uuid4()} {random.random():.12f}]\n"
            chunk = json.load(post("/detokenize", {"model": args.model, "tokens": ids[start:start + target - 40]}))["prompt"]
            prompt = salt + chunk + "\n\nReply with OK."
            m0 = metrics()
            o = stream(prompt, 8, {})
            m1 = metrics()
            ttft = o["first"] - o["t0"]
            rate = o["prompt_tokens"] / ttft
            hits = m1.get("vllm:prefix_cache_hits_total", 0) - m0.get("vllm:prefix_cache_hits_total", 0)
            rates.append(rate)
            results["prefill"].append({"target": target, "prompt_tokens": o["prompt_tokens"], "ttft": round(ttft, 3),
                                       "rate": round(rate, 1), "cache_hit_tokens": hits, "reply": o["text"][:40],
                                       "busy_before": {"running": m0.get("vllm:num_requests_running", 0)}})
            print(f"prefill {o['prompt_tokens']:6d} tok  ttft {ttft:6.2f}s  {rate:7.1f} tok/s  cache hits {hits:.0f}", flush=True)
            save()
    results["prefill_median"] = round(statistics.median(rates), 1)
    print(f"cold prefill median {statistics.median(rates):.1f} tok/s", flush=True)

results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
