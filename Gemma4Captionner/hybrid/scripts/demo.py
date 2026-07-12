"""One-command demo launcher.

    python scripts/demo.py

Loads API keys from the machine's environment / .env (never hardcoded), then:
  - serves the caption tester (upload or URL, video shown next to captions)
    on http://127.0.0.1:8799
  - serves the demo pages (model comparison with the source clips embedded,
    captions on the official jury clips) on http://127.0.0.1:8788
  - opens both in the default browser
"""
from __future__ import annotations

import functools
import http.server
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")  # machine-provided keys; nothing is baked in the repo

missing = [k for k in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "FIREWORKS_API_KEY")
           if not os.environ.get(k)]
if missing:
    print(f"note: {', '.join(missing)} not set - the tester will degrade to whatever provider is available")

from app.webapp import app  # noqa: E402  (needs env loaded first)


def serve_docs() -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    http.server.ThreadingHTTPServer(("127.0.0.1", 8788), handler).serve_forever()


threading.Thread(target=serve_docs, daemon=True).start()
print("demo pages  : http://127.0.0.1:8788/docs/comparison.html  (models side by side + source clips)")
print("              http://127.0.0.1:8788/docs/official.html    (captions on the jury clips)")
print("caption ANY video: http://127.0.0.1:8799  (paste a URL or upload a file)")
webbrowser.open("http://127.0.0.1:8788/docs/comparison.html")
webbrowser.open("http://127.0.0.1:8799")
app.run(host="127.0.0.1", port=8799, threaded=True)
