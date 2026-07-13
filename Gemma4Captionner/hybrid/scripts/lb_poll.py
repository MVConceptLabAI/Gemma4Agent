# Leaderboard poller v2: reassembles Next.js __next_f streamed chunks.
import json, re, sys, urllib.request

URL = "https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/live?track=2"
html = ""
for _ in range(4):  # truncated responses happen; retry until the payload is complete
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    html = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")
    if "submissionId" in html or "submissionId" in html.replace(chr(92)+'"', '"'):
        break
    import time; time.sleep(5)

# reassemble streamed payload: self.__next_f.push([1,"...chunk..."])
chunks = []
for m in re.finditer(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html):
    try:
        chunks.append(json.loads(m.group(1)))  # unescape the JS string literal
    except Exception:
        pass
stream = "".join(chunks)
candidates = [stream, stream.replace('\\"', '"'), html.replace('\\"', '"')]

entries = []
for txt in candidates:
    for m in re.finditer(r'\{"submissionId":"[^"]+".*?"evaluatedAt":"?[^,}]*"?\}', txt):
        try:
            entries.append(json.loads(m.group(0)))
        except Exception:
            pass
    if entries:
        break

seen = {}
for e in entries:
    seen[e.get("submissionId")] = e
entries = list(seen.values())

ours = next((e for e in entries if e.get("teamSlug") == "mvconceptlab"), None)
scored = sorted((e for e in entries if isinstance(e.get("score"), (int, float)) and 0 < e["score"] < 1.0),
                key=lambda e: -e["score"])
top = " | ".join(f"#{i+1} {e['teamName']} {e['score']:.4f}" for i, e in enumerate(scored[:3]))
if ours:
    line = (f"MV rank={ours.get('rank')} score={ours.get('score')} status={ours.get('status') or ours.get('failureReason')} "
            f"resub={str(ours.get('lastResubmittedAt'))[11:16]} eval={str(ours.get('evaluatedAt'))[11:16]} || TOP: {top}")
else:
    line = f"MV NOT FOUND ({len(entries)} entries) || TOP: {top}"
print(line)
