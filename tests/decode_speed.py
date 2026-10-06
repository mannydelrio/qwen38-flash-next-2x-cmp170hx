"""Decode-speed check with short prompts (prefill is negligible): prose and code, 1 stream and 4 streams.
Prose drafts are accepted less often than code, so it shows raw per-step speed better than HumanEval."""
import json, os, time, urllib.request, concurrent.futures
KEY = os.environ.get("VLLM_API_KEY", "")
URL = "http://127.0.0.1:18084"
PROMPTS = {
    "prose": ["Write a detailed short story about a lighthouse keeper who finds a message in a bottle.",
              "Explain the history of the printing press and its effects on European society.",
              "Describe a week-long road trip through the American Southwest, day by day."],
    "code": ["Write a Python module implementing an LRU cache class with get/put, type hints and docstrings, plus unit tests.",
             "Write a TypeScript Express server with CRUD routes for a todo list stored in memory, with validation.",
             "Write a Python script that parses an nginx access log and prints the top 10 IPs, paths and status codes."],
}

def metrics():
    req = urllib.request.Request(URL + "/metrics", headers={"Authorization": "Bearer " + KEY})
    out = {}
    for line in urllib.request.urlopen(req, timeout=5).read().decode().splitlines():
        for k in ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(k + "{"):
                out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return out

def go(prompt):
    body = {"model": "tinybeast-flashnext-uncensored", "max_tokens": 512, "ignore_eos": True, "temperature": 0.7,
            "top_p": 0.95, "seed": 7, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return d["usage"]["completion_tokens"], time.time() - t

for kind, prompts in PROMPTS.items():
    m0 = metrics()
    single = [go(p) for p in prompts]
    tps1 = sum(t for t, _ in single) / sum(s for _, s in single)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        quad = list(pool.map(go, (prompts * 2)[:4]))
    agg4 = sum(t for t, _ in quad) / (time.time() - t0)
    m1 = metrics()
    dr = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    ac = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    print(f"decode {kind:5s}: 1 stream {tps1:6.1f} tok/s | 4 streams {agg4:6.1f} tok/s total | draft acceptance "
          f"{100 * ac / dr if dr else 0:.0f}%", flush=True)
