"""Long-context retrieval check: hide one fact at a given depth in a word-salad haystack, ask for it back.
Usage: python3 needle.py TOKENS DEPTH [TOKENS DEPTH ...]"""
import json, os, random, sys, time, urllib.request
KEY = os.environ.get("VLLM_API_KEY", "")
WORDS = ("river stone market lantern orbit copper meadow signal harbor velvet summit cipher marble thunder "
         "garden pixel quarry violet engine forest canyon ember glacier harvest island jigsaw kernel ladder "
         "mirror nectar oyster prairie quartz ribbon saddle timber umbrella valley walnut yonder zephyr anchor "
         "beacon candle dolphin falcon granite horizon iris jasmine kettle lagoon maple nimbus opal pebble").split()
args = sys.argv[1:]
for a in range(0, len(args), 2):
    tokens, depth = int(args[a]), float(args[a + 1])
    rnd = random.Random(tokens)
    code = f"{rnd.randint(100, 999)}-{rnd.choice(['ZEBRA', 'OTTER', 'MANGO', 'COMET'])}-{rnd.randint(10, 99)}"
    words = [rnd.choice(WORDS) for _ in range(int(tokens / 1.35))]
    needle = f" The secret vault code is {code}. Remember it. "
    cut = int(len(words) * depth)
    hay = " ".join(words[:cut]) + needle + " ".join(words[cut:])
    q = hay + "\n\nWhat is the secret vault code mentioned in the text above? Reply with only the code."
    body = {"model": "tinybeast-flashnext-uncensored", "messages": [{"role": "user", "content": q}], "max_tokens": 30,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:18084/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        ans = (d["choices"][0]["message"].get("content") or "").strip()
        pt = d["usage"]["prompt_tokens"]; s = time.time() - t
        print(f"{pt:>9,} tokens, depth {depth:.2f}: {'PASS' if code in ans else 'FAIL'} | expected {code} got {ans[:40]!r} | {s:.0f}s ({pt/s:,.0f} tok/s)", flush=True)
    except urllib.error.HTTPError as e:
        print(f"~{tokens:,} tokens: HTTP {e.code} {e.read().decode()[:200]}", flush=True)
    except Exception as e:
        print(f"~{tokens:,} tokens: ERROR {str(e)[:200]}", flush=True)
