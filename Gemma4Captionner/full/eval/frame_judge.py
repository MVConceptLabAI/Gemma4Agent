# Official-rubric proxy judge: scores captions against REAL frames.
#   PYTHONPATH=. python eval/frame_judge.py --results out/x.json --tasks data/sample_tasks.json
# Guide wording: accuracy = "how faithfully the caption reflects the video
# content"; style = "how well the caption matches the requested tone".
import argparse, asyncio, base64, json, math, os, subprocess, sys, tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
import httpx

sys.stdout.reconfigure(encoding="utf-8")
env_path = Path(".env")
if env_path.is_file():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            os.environ.setdefault(
                "OPENROUTER_API_KEY", line.partition("=")[2].strip()
            )

JUDGE = os.environ.get("JUDGE_MODEL", "google/gemini-3.1-pro-preview")
STYLE_DEFS = {
    "formal": "Professional, objective, factual tone",
    "sarcastic": "Dry, ironic, lightly mocking",
    "humorous_tech": "Funny, with technology or programming references",
    "humorous_non_tech": "Funny, everyday humour with no technical jargon",
}
RUBRIC = (
    "You are the LLM-Judge of a video-captioning contest. You see frames sampled "
    "from ONE video clip, a requested style, and a caption. Score two dimensions:\n"
    "1. accuracy (0-1): how faithfully the caption reflects the video content - "
    "penalize any claim not supported by the frames; reward correct, specific coverage.\n"
    "2. style (0-1): how well the caption matches the requested tone.\n"
    'Return STRICT JSON: {"accuracy": <float>, "style": <float>, "wrong_claims": ["..."]}'
)


def frames_b64(url: str, n: int = 6) -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        vp = Path(td) / "c.mp4"
        direct_path = Path(url)
        if direct_path.is_file():
            video_path = direct_path
        elif urlparse(url).scheme == "file":
            parsed = urlparse(url)
            local_path = url2pathname(unquote(parsed.path))
            if parsed.netloc:
                local_path = f"//{parsed.netloc}{local_path}"
            video_path = Path(local_path)
            if not video_path.is_file():
                raise FileNotFoundError(f"local video does not exist: {video_path}")
        else:
            for attempt in range(3):
                try:
                    subprocess.run(["curl", "-sL", url, "-o", str(vp)], check=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt == 2:
                        raise
                    print(f"  video download retry {attempt + 2}/3: {url}", file=sys.stderr)
            video_path = vp
        out: list[str] = []
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)]).strip() or 30)
        for i in range(1, n + 1):
            fp = Path(td) / f"f{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                            "-ss", str(round(dur * i / (n + 1), 2)), "-i", str(video_path),
                            "-frames:v", "1", "-vf", "scale=640:-2", str(fp)], check=True)
            out.append(base64.b64encode(fp.read_bytes()).decode())
        return out


def _skipped_score(reason: str) -> dict:
    return {
        "accuracy": float("nan"),
        "style": float("nan"),
        "wrong_claims": [],
        "skipped": reason,
    }


async def judge_one(client, frames, style, caption):
    content = [{"type": "text", "text": f"Requested style: {style} = {STYLE_DEFS[style]}\n\nCaption:\n{caption}\n\nFrames:"}]
    for b in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    for attempt in range(3):
        try:
            r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json={"model": JUDGE, "messages": [{"role": "system", "content": RUBRIC},
                      {"role": "user", "content": content}], "max_tokens": 3000, "temperature": 0.0})
        except httpx.HTTPError as exc:
            if attempt == 2:
                reason = f"network error after 3 attempts: {type(exc).__name__}: {exc}"
                print(f"  judge skipped: {reason}", file=sys.stderr)
                return _skipped_score(reason)
            await asyncio.sleep(2)
            continue
        try:
            t = r.json()["choices"][0]["message"]["content"]
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception:  # noqa: BLE001 - empty/reasoning-truncated reply: retry
            if attempt == 2:
                print(f"  judge unparseable ({r.status_code}): {r.text[:120]}", file=sys.stderr)
                return _skipped_score(f"unparseable response after 3 attempts (HTTP {r.status_code})")
            await asyncio.sleep(2)


async def evaluate_results(results, tasks, client, frame_loader=frames_b64, judge_fn=judge_one):
    rows, total_accuracy, total_style, count = [], 0.0, 0.0, 0
    for result in results:
        task_id = result["task_id"]
        captions = result["captions"]
        try:
            frames = frame_loader(tasks[task_id])
        except Exception as exc:  # noqa: BLE001 - one bad clip must not abort the comparison
            reason = f"frame extraction failed: {type(exc).__name__}: {exc}"
            for style in captions:
                rows.append({"task_id": task_id, "style": style, "judge": _skipped_score(reason)})
            continue

        judgments = await asyncio.gather(
            *[judge_fn(client, frames, style, caption) for style, caption in captions.items()],
            return_exceptions=True,
        )
        for style, judgment in zip(captions, judgments):
            if isinstance(judgment, BaseException):
                reason = f"judge failed: {type(judgment).__name__}: {judgment}"
                score = _skipped_score(reason)
            elif not isinstance(judgment, dict):
                score = _skipped_score(f"judge returned {type(judgment).__name__}, expected object")
            else:
                try:
                    accuracy = float(judgment["accuracy"])
                    style_score = float(judgment["style"])
                except (KeyError, TypeError, ValueError) as exc:
                    score = _skipped_score(f"invalid judge score: {type(exc).__name__}: {exc}")
                else:
                    if math.isfinite(accuracy) and math.isfinite(style_score):
                        score = dict(judgment, accuracy=accuracy, style=style_score)
                        total_accuracy += accuracy
                        total_style += style_score
                        count += 1
                    else:
                        score = dict(judgment)
                        score["skipped"] = score.get("skipped", "non-finite judge score")
            rows.append({"task_id": task_id, "style": style, "judge": score})
    return rows, total_accuracy, total_style, count


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tasks", required=True)
    a = ap.parse_args()
    tasks = {t["task_id"]: t["video_url"] for t in json.load(open(a.tasks))}
    results = json.load(open(a.results, encoding="utf-8"))
    async with httpx.AsyncClient(timeout=120) as client:
        rows, tot_a, tot_s, n = await evaluate_results(results, tasks, client)
    for row in rows:
        task_id, style, judgment = row["task_id"], row["style"], row["judge"]
        if "skipped" in judgment:
            print(f"{task_id:10s} {style:18s} SKIP {judgment['skipped']}")
            continue
        wrong = ("  WRONG: " + "; ".join(judgment.get("wrong_claims", [])[:2])) if judgment.get("wrong_claims") else ""
        print(f"{task_id:10s} {style:18s} acc={judgment['accuracy']:.2f} style={judgment['style']:.2f}{wrong}")
    if n:
        print(f"\nMEAN accuracy={tot_a/n:.3f} style={tot_s/n:.3f} FINAL={(tot_a+tot_s)/(2*n):.3f} (n={n})")
    else:
        print("\nNO FINITE SCORES: every clip/judgment was skipped")

if __name__ == "__main__":
    asyncio.run(main())
