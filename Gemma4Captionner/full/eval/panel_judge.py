"""Severe 2-axis judge PANEL. Reports per-style accuracy vs style so we can see
which axis/style actually drags - instead of guessing.

  PYTHONPATH=. python eval/panel_judge.py --results out/x.json --tasks data/official_new12.json
"""
import argparse, asyncio, base64, json, os, subprocess, sys, tempfile, statistics
from pathlib import Path
import httpx

sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env"):
    if line.startswith("OPENROUTER_API_KEY="):
        os.environ.setdefault("OPENROUTER_API_KEY", line.partition("=")[2].strip())
K = os.environ["OPENROUTER_API_KEY"]

JUDGES = ["google/gemini-3.1-pro-preview", "openai/gpt-5.5"]
STYLE_DEFS = {
    "formal": "Professional, objective, factual tone. No jokes, no first/second person.",
    "sarcastic": "Dry, ironic, lightly mocking. No technology words.",
    "humorous_tech": "Funny, WITH technology/programming references tied to what's visible.",
    "humorous_non_tech": "Funny, everyday humour, NO technical jargon.",
}
RUBRIC = (
    "You are a STRICT judge in a video-captioning contest. Be harsh and discriminating - "
    "most captions should NOT get 1.0. You see frames from ONE clip, a target style, and a caption.\n"
    "Score TWO axes independently:\n"
    "- accuracy (0.0-1.0): does every claim match the frames? Deduct 0.15 for each detail that is "
    "wrong, invented, or unsupported (fake colors, fake signs, fake motion, time-lapse cliches). "
    "Reward correct SPECIFIC coverage; a vague-but-safe caption caps at ~0.7.\n"
    "- style (0.0-1.0): how strongly and cleanly does it hit the TARGET tone? A caption that is "
    "merely descriptive with a light garnish of tone scores ~0.6; only a caption whose whole voice "
    "embodies the style scores >0.9. Penalize style-bleed (tech words in a no-tech style, jokes in formal).\n"
    'Return STRICT JSON: {"accuracy": <float>, "style": <float>, "flaw": "<=8 words"}'
)


def frames_b64(url, n=6):
    with tempfile.TemporaryDirectory() as td:
        vp = Path(td) / "c.mp4"
        for _ in range(3):
            try:
                subprocess.run(["curl", "-sL", "--max-time", "60", url, "-o", str(vp)], check=True)
                break
            except Exception:
                pass
        out = []
        try:
            dur = float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(vp)]).strip() or 30)
        except Exception:
            dur = 30
        for i in range(1, n + 1):
            fp = Path(td) / f"f{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                            "-ss", str(round(dur * i / (n + 1), 2)), "-i", str(vp),
                            "-frames:v", "1", "-vf", "scale=512:-2", str(fp)], check=False)
            if fp.exists():
                out.append(base64.b64encode(fp.read_bytes()).decode())
        return out


async def judge(client, model, frames, style, caption):
    content = [{"type": "text", "text": f"Target style: {style} = {STYLE_DEFS[style]}\n\nCaption:\n{caption}\n\nFrames:"}]
    for b in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    for _ in range(3):
        try:
            r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {K}"},
                json={"model": model, "messages": [{"role": "system", "content": RUBRIC},
                      {"role": "user", "content": content}], "max_tokens": 2500, "temperature": 0.0})
            t = r.json()["choices"][0]["message"]["content"]
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception:
            await asyncio.sleep(2)
    return {"accuracy": float("nan"), "style": float("nan"), "flaw": "judge-fail"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tasks", required=True)
    a = ap.parse_args()
    tasks = {t["task_id"]: t["video_url"] for t in json.load(open(a.tasks))}
    results = json.load(open(a.results, encoding="utf-8"))
    per_style = {s: {"acc": [], "sty": []} for s in STYLE_DEFS}
    async with httpx.AsyncClient(timeout=150) as client:
        for r in results:
            fb = frames_b64(tasks[r["task_id"]])
            for s, c in r["captions"].items():
                verdicts = await asyncio.gather(*[judge(client, m, fb, s, c) for m in JUDGES])
                accs = [v["accuracy"] for v in verdicts if v["accuracy"] == v["accuracy"]]
                stys = [v["style"] for v in verdicts if v["style"] == v["style"]]
                if accs: per_style[s]["acc"].append(statistics.mean(accs))
                if stys: per_style[s]["sty"].append(statistics.mean(stys))
                flaws = "; ".join(v.get("flaw", "") for v in verdicts)
                print(f"{r['task_id']:12s} {s:18s} acc={statistics.mean(accs) if accs else 0:.2f} sty={statistics.mean(stys) if stys else 0:.2f}  {flaws[:70]}")
    print("\n===== PER-STYLE MEANS =====")
    ta, ts = [], []
    for s, d in per_style.items():
        am = statistics.mean(d["acc"]) if d["acc"] else 0
        sm = statistics.mean(d["sty"]) if d["sty"] else 0
        ta += d["acc"]; ts += d["sty"]
        print(f"  {s:18s} accuracy={am:.3f}  style={sm:.3f}")
    print(f"\nOVERALL accuracy={statistics.mean(ta):.3f}  style={statistics.mean(ts):.3f}  FINAL={(statistics.mean(ta)+statistics.mean(ts))/2:.3f}")

asyncio.run(main())
