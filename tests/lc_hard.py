"""Harder long-context test: real source code as the haystack, 10 planted facts at fixed depths, 8 questions in one
request (retrieval at 3-97% depth, a superseded value, a two-hop join, a three-number sum, and an unanswerable trap).
  python3 lc_hard.py LABEL TOKENS [TOKENS ...]       e.g. python3 lc_hard.py native 64000 128000 245000"""
import json, os, re, sys, time, uuid, urllib.request
from pathlib import Path

KEY = os.environ.get("VLLM_API_KEY", "")
BASE = "http://127.0.0.1:18084"
SRC = Path(os.environ.get("HAYSTACK_DIR", "vllm-backport/vllm"))

FACTS = [  # (depth, sentence)
    (0.03, "Operations note: the maintenance password for the Lantern server is violet-harbor-58."),
    (0.15, "Operations note: the ingestion queue is owned by the team called Kestrel."),
    (0.25, "Operations note: Project Orchid is deployed in region eu-north-7."),
    (0.35, "Operations note: Warehouse Delta received 42 pallets on Monday."),
    (0.40, "Operations note: the max retry budget for the ingestion queue is 17."),
    (0.50, "Operations note: the on-call engineer for the billing service in March is Priya Raman."),
    (0.60, "Operations note: Warehouse Delta received 19 pallets on Wednesday."),
    (0.85, "Operations note: team Kestrel's Slack channel is #kestrel-firefighting."),
    (0.90, "Operations note: Warehouse Delta received 27 pallets on Friday."),
    (0.92, "Operations note: update - the max retry budget for the ingestion queue was raised from 17 to 23."),
    (0.97, "Operations note: the backup window for the analytics cluster starts at 02:40 UTC."),
]
QUESTIONS = [
    ("What is the maintenance password for the Lantern server?", lambda a: "violet-harbor-58" in a),
    ("In which region is Project Orchid deployed?", lambda a: "eu-north-7" in a),
    ("Who is the on-call engineer for the billing service in March?", lambda a: "priya" in a),
    ("What is the current max retry budget for the ingestion queue?", lambda a: "23" in a),
    ("When does the backup window for the analytics cluster start?", lambda a: "02:40" in a or "2:40" in a),
    ("What is the Slack channel of the team that owns the ingestion queue?", lambda a: "kestrel-firefighting" in a),
    ("How many pallets did Warehouse Delta receive in total that week?", lambda a: re.search(r"\b88\b", a) is not None),
    ("In which region is Project Willow deployed?",
     lambda a: any(w in a for w in ("not mention", "not stated", "not specified", "no information", "doesn't", "does not",
                                     "isn't", "is not", "not found", "unknown", "not provided", "no mention"))
               and not re.search(r"\b[a-z]{2}-[a-z]+-\d\b", a)),
]


def corpus(chars):
    out, n = [], 0
    for f in sorted(SRC.rglob("*.py")):
        t = f.read_text(errors="ignore")
        out.append(f"\n### file: {f.relative_to(SRC)}\n{t}")
        n += len(out[-1])
        if n >= chars:
            break
    return "".join(out)[:chars]


def tokenize(text):
    req = urllib.request.Request(BASE + "/tokenize", json.dumps({"model": "tinybeast-flashnext-uncensored", "prompt": text}).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["count"]


def build(tokens, ratio):
    hay = corpus(int(tokens * ratio))
    lines = hay.split("\n")
    for depth, sentence in sorted(FACTS, reverse=True):  # insert from the end so earlier indices stay valid
        lines.insert(int(len(lines) * depth), sentence)
    qs = "\n".join(f"{i + 1}. {q}" for i, (q, _) in enumerate(QUESTIONS))
    return ("Below is a large dump of source code with some operations notes mixed in.\n\n" + "\n".join(lines) +
            "\n\nUsing only the text above, answer each question in one short line, numbered 1-8. "
            "If the text does not contain the answer, say so.\n" + qs)


def ask(prompt):
    body = {"model": "tinybeast-flashnext-uncensored", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "cache_salt": str(uuid.uuid4())}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return d["choices"][0]["message"]["content"] or "", d["usage"]["prompt_tokens"], time.time() - t


def main():
    label, sizes = sys.argv[1], [int(x) for x in sys.argv[2:]]
    sample = corpus(200_000)
    ratio = len(sample) / tokenize(sample)  # chars per token for this corpus
    results = []
    for target in sizes:
        prompt = build(target, ratio)
        n = tokenize(prompt)
        while n > target:  # trim to the target exactly-ish
            prompt = build(int(target * target / n) - 500, ratio)
            n = tokenize(prompt)
        try:
            ans, pt, sec = ask(prompt)
        except Exception as e:
            print(f"[{label}] {target:,}: ERROR {str(e)[:200]}", flush=True)
            continue
        lines = {int(m.group(1)): m.group(2).strip().lower()
                 for m in re.finditer(r"^\s*(\d+)[.)]\s*(.*)$", ans, re.M)}
        score = [bool(chk(lines.get(i + 1, ""))) for i, (_, chk) in enumerate(QUESTIONS)]
        print(f"[{label}] {pt:,} tokens, {sec:.0f}s: {sum(score)}/8  " +
              " ".join(f"Q{i + 1}{'+' if s else '-'}" for i, s in enumerate(score)), flush=True)
        for i in range(1, 9):
            print(f"      {i}. {lines.get(i, '(missing)')[:110]}", flush=True)
        results.append({"label": label, "tokens": pt, "seconds": sec, "score": score, "answer": ans})
    out = Path(f"lc_hard-{label}.json")
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
