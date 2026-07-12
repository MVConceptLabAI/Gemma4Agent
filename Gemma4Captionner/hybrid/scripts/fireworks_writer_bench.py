# Benchmark all Fireworks text models as the STYLE writer, same facts, real prompts.
# Usage: PYTHONPATH=. python scripts/fireworks_writer_bench.py
import os, json, time, sys
import httpx

sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env"):
    if line.startswith("FIREWORKS_API_KEY="):
        os.environ["FIREWORKS_API_KEY"] = line.partition("=")[2].strip()
os.environ["MAX_CAPTION_CHARS"] = "1600"

from app.prompts import STYLE_PROMPTS
from app.models import caption_passes_style_filter
from app.pipeline import _extract_final_caption

K = os.environ["FIREWORKS_API_KEY"]

# v1 traffic clip — details verified by the vision audit (ensemble run)
FACTS = {
    "summary": "A high-angle time-lapse of a busy multi-lane urban boulevard in autumn golden-hour light",
    "setting": "Korean city boulevard lined with yellow ginkgo trees; hazy mountains behind",
    "subjects": ["dense car traffic in right lanes", "freer left lanes", "a blue city bus", "white box trucks"],
    "actions": ["vehicles streak with motion blur", "right lanes queue slowly", "bus crosses left to right"],
    "visual_details": [
        "rows of yellow-leafed ginkgo trees on both sides",
        "green traffic lights over the intersection",
        "blue circular U-turn sign",
        "signs reading TAXPARK INSURANCE and KOREA ILLIES ENGINEERING",
        "a green Starbucks logo at street level",
        "at least eight tall apartment towers in the background",
        "white and pink vertical banners on the right guardrail",
        "long shadows from warm low sunlight",
    ],
    "mood": "busy, golden, autumnal",
    "tech_visible": False,
}
STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
MODELS = ["gpt-oss-120b", "deepseek-v4-pro", "kimi-k2p6", "kimi-k2p5", "glm-5p1", "glm-5p2"]


def strip_think(text: str) -> str:
    # reasoning models may leak <think>...</think> or plain preamble into content
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return _extract_final_caption(text)


def call(model: str, style: str) -> tuple[str, float, str]:
    system, few_shots, user_tpl = STYLE_PROMPTS[style]
    msgs = [{"role": "system", "content": system}]
    for ex_f, ex_c in few_shots:
        msgs += [{"role": "user", "content": user_tpl.format(facts=ex_f)},
                 {"role": "assistant", "content": ex_c}]
    msgs.append({"role": "user", "content": user_tpl.format(facts=json.dumps(FACTS))})
    t0 = time.time()
    r = httpx.post("https://api.fireworks.ai/inference/v1/chat/completions",
                   headers={"Authorization": f"Bearer {K}"},
                   json={"model": f"accounts/fireworks/models/{model}", "messages": msgs,
                         "max_tokens": 1600, "temperature": 0.7}, timeout=120)
    dt = time.time() - t0
    if r.status_code != 200:
        return "", dt, f"HTTP {r.status_code}"
    raw = r.json()["choices"][0]["message"].get("content") or ""
    cap = strip_think(raw)
    leak = "LEAK" if any(w in cap.lower()[:120] for w in ("the user", "we are asked", "let me", "i need to")) else ""
    return cap, dt, leak


results = {}
for m in MODELS:
    rows = []
    for s in STYLES:
        try:
            cap, dt, err = call(m, s)
        except Exception as e:  # noqa: BLE001
            cap, dt, err = "", 0.0, str(e)[:40]
        ok = bool(cap) and caption_passes_style_filter(s, cap) and not err
        rows.append({"style": s, "ok": ok, "dt": round(dt, 1), "err": err,
                     "len": len(cap), "cap": cap[:400]})
        print(f"{m:16s} {s:18s} ok={ok} {dt:5.1f}s {err} :: {cap[:90]}")
    results[m] = rows

json.dump(results, open("out/fireworks_writer_bench.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nSUMMARY (pass/4, avg latency):")
for m, rows in results.items():
    npass = sum(r["ok"] for r in rows)
    avg = sum(r["dt"] for r in rows) / len(rows)
    print(f"  {m:18s} {npass}/4 pass  {avg:5.1f}s avg")
