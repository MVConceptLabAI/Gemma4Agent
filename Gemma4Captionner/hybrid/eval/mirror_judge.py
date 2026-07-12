"""Official judge MIRROR (from amd_hackathon_kit/06_EVAL_HARNESS.md).

The real judge compares a caption to a fixed TEXT ground-truth scene with a
graded rubric (one hallucination -> 0.5), NOT to frames. This reproduces that.

  # 1. build ground truth once (reusable):
  PYTHONPATH=. python eval/mirror_judge.py --build-gt --tasks data/official_new12.json --ids 11785757,12471596,2697636,31948459
  # 2. score a results file:
  PYTHONPATH=. python eval/mirror_judge.py --results out/v14_clean.json
"""
import argparse, asyncio, base64, json, os, subprocess, sys, tempfile, statistics
from pathlib import Path
import httpx

sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env"):
    if line.startswith("OPENROUTER_API_KEY="):
        os.environ.setdefault("OPENROUTER_API_KEY", line.partition("=")[2].strip())
K = os.environ["OPENROUTER_API_KEY"]
GT_PATH = "out/scene_gt.json"
JUDGE_MODEL = os.environ.get("MIRROR_JUDGE_MODEL", "openai/gpt-5.5")
GT_MODEL = os.environ.get("GT_MODEL", "google/gemini-3.1-pro-preview")

STYLE_DEFS = {
    "formal": "Professional, objective, and factual tone.",
    "sarcastic": "Dry, ironic, and lightly mocking.",
    "humorous_tech": "Funny, with technology or programming references.",
    "humorous_non_tech": "Funny, everyday humor with no technical jargon.",
}

JUDGE_PROMPT = """You are an impartial expert evaluator scoring video captions on two dimensions.

VIDEO SCENE (ground truth, human-annotated):
{scene}

STYLE REQUESTED: {style}
STYLE DEFINITION: {style_definition}

CAPTION TO EVALUATE:
{caption}

RUBRIC:

1. CAPTION ACCURACY (0.0 to 1.0):
   How faithfully does the caption reflect the actual video content?
   - 1.0: All key elements (subjects, actions, setting) accurately described. No hallucinations.
   - 0.75: Mostly accurate, minor omissions or slight embellishments.
   - 0.5: Correct on major points but misses 1-2 important elements OR includes a minor hallucination.
   - 0.25: Partially correct, significant misses or fabrications.
   - 0.0: Wrong subject / setting / action entirely, or complete hallucination.

2. STYLE MATCH (0.0 to 1.0):
   How well does the caption match the requested style definition?
   - 1.0: Perfect embodiment of the style. Native, natural, effortless.
   - 0.75: Clearly in the style but with minor lapses.
   - 0.5: Recognizably attempting the style but inconsistent.
   - 0.25: Only weak style signals.
   - 0.0: Style is wrong or absent.

REQUIRED OUTPUT (strict JSON, no other text):
{{"accuracy": <float 0-1>, "style_match": <float 0-1>, "accuracy_reason": "<one sentence>", "style_reason": "<one sentence>"}}"""


def _frames(url, n=8):
    with tempfile.TemporaryDirectory() as td:
        vp = Path(td) / "c.mp4"
        subprocess.run(["curl", "-sL", "--max-time", "90", url, "-o", str(vp)], check=True)
        try:
            dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "csv=p=0", str(vp)]).strip() or 30)
        except Exception:
            dur = 30
        out = []
        for i in range(1, n + 1):
            fp = Path(td) / f"f{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss",
                str(round(dur * i / (n + 1), 2)), "-i", str(vp), "-frames:v", "1",
                "-vf", "scale=768:-2", str(fp)], check=False)
            if fp.exists():
                out.append(base64.b64encode(fp.read_bytes()).decode())
        return out


async def _post(client, model, messages, mt=1500):
    for _ in range(3):
        try:
            r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {K}"},
                json={"model": model, "messages": messages, "max_tokens": mt, "temperature": 0.0})
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            await asyncio.sleep(2)
    return ""


async def build_gt(tasks_file, ids):
    tasks = {t["task_id"]: t["video_url"] for t in json.load(open(tasks_file))}
    gt = json.load(open(GT_PATH)) if os.path.exists(GT_PATH) else {}
    async with httpx.AsyncClient(timeout=180) as client:
        for tid in ids:
            if tid in gt:
                print(f"{tid}: cached"); continue
            fb = _frames(tasks[tid])
            content = [{"type": "text", "text": "Write an exhaustive, strictly factual, neutral ground-truth description of this video clip for use as a scoring reference: every subject, action, setting, notable object, colour and any change over time that is CLEARLY visible. No speculation, no style, no adjectives of opinion. 4-8 sentences."}]
            for b in fb:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
            gt[tid] = (await _post(client, GT_MODEL, [{"role": "user", "content": content}], 900)).strip()
            print(f"{tid}: {gt[tid][:110]}")
    json.dump(gt, open(GT_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {GT_PATH} ({len(gt)} clips)")


async def score(results_file):
    gt = json.load(open(GT_PATH, encoding="utf-8"))
    results = json.load(open(results_file, encoding="utf-8"))
    A, S = [], []
    per_style = {s: {"a": [], "s": []} for s in STYLE_DEFS}
    async with httpx.AsyncClient(timeout=120) as client:
        for r in results:
            scene = gt.get(r["task_id"])
            if not scene:
                print(f"!! no GT for {r['task_id']} - skip"); continue
            for style, cap in r["captions"].items():
                prompt = JUDGE_PROMPT.format(scene=scene, style=style,
                    style_definition=STYLE_DEFS[style], caption=cap)
                txt = await _post(client, JUDGE_MODEL, [{"role": "user", "content": prompt}], 800)
                try:
                    o = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
                    a, s = float(o["accuracy"]), float(o["style_match"])
                except Exception:
                    print(f"  parse-fail {r['task_id']}/{style}: {txt[:60]}"); continue
                A.append(a); S.append(s)
                per_style[style]["a"].append(a); per_style[style]["s"].append(s)
                print(f"{r['task_id']:12s} {style:18s} acc={a:.2f} sty={s:.2f}  {o.get('accuracy_reason','')[:55]}")
    print("\n===== MIRROR (official rubric) =====")
    for st, d in per_style.items():
        am = statistics.mean(d["a"]) if d["a"] else 0
        sm = statistics.mean(d["s"]) if d["s"] else 0
        print(f"  {st:18s} accuracy={am:.3f}  style={sm:.3f}")
    if A:
        print(f"\nOVERALL accuracy={statistics.mean(A):.3f}  style={statistics.mean(S):.3f}  FINAL={(statistics.mean(A)+statistics.mean(S))/2:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-gt", action="store_true")
    ap.add_argument("--tasks")
    ap.add_argument("--ids")
    ap.add_argument("--results")
    a = ap.parse_args()
    if a.build_gt:
        asyncio.run(build_gt(a.tasks, a.ids.split(",")))
    else:
        asyncio.run(score(a.results))


if __name__ == "__main__":
    main()
