import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from aireceipes.catalog import Catalog
from aireceipes.runner import run_recipe


def write_recipe(root: Path) -> Path:
    recipe_dir = root / "inference"
    recipe_dir.mkdir(parents=True)
    recipe_path = recipe_dir / "echo-smoke.toml"
    recipe_path.write_text(
        """
id = "inference.echo-smoke"
category = "inference"
name = "Echo chat latency smoke"
description = "Deterministic smoke benchmark for validating the recipe runner."
tags = ["smoke", "text", "chat"]

[classification]
task = "chat-completions"
modality = "text"
size = "smoke"

[runtime]
adapter = "echo"

[parameters]
repetitions = 1

[dataset]
prompts = [
  "Say HELLO in one word.",
  "Return the word KPI."
]

[kpis]
metrics = ["success_rate", "latency_ms_avg", "output_chars_avg"]
""".strip(),
        encoding="utf-8",
    )
    return recipe_path


def test_catalog_discovers_classified_recipes(tmp_path):
    write_recipe(tmp_path)

    recipes = Catalog(tmp_path).list()

    assert [recipe.id for recipe in recipes] == ["inference.echo-smoke"]
    assert recipes[0].category == "inference"
    assert recipes[0].classification["task"] == "chat-completions"
    assert "smoke" in recipes[0].tags


def test_echo_recipe_run_writes_metrics_json(tmp_path):
    write_recipe(tmp_path / "recipes")
    output_dir = tmp_path / "runs"

    result = run_recipe("inference.echo-smoke", recipes_dir=tmp_path / "recipes", output_dir=output_dir)

    metrics_path = result.run_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["recipe_id"] == "inference.echo-smoke"
    assert metrics["category"] == "inference"
    assert metrics["prompt_count"] == 2
    assert metrics["success_count"] == 2
    assert metrics["error_count"] == 0
    assert metrics["kpis"]["success_rate"] == 1.0
    assert metrics["kpis"]["latency_ms_avg"] >= 0
    assert metrics["kpis"]["output_chars_avg"] > 0


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "payload": payload,
            }
        )
        response = {
            "choices": [
                {
                    "message": {
                        "content": f"fake:{payload['messages'][-1]['content']}",
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        return None


def test_openai_compatible_recipe_posts_chat_completions_and_writes_outputs(tmp_path, monkeypatch):
    _OpenAICompatibleHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("AIRECEIPES_TEST_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    try:
        recipes_dir = tmp_path / "recipes"
        recipe_dir = recipes_dir / "inference"
        recipe_dir.mkdir(parents=True)
        (recipe_dir / "local-chat-smoke.toml").write_text(
            '''
id = "inference.local-chat-smoke"
category = "inference"
name = "Local OpenAI-compatible chat smoke"
description = "Calls an OpenAI-compatible /v1/chat/completions endpoint."
tags = ["smoke", "text", "chat", "openai-compatible"]

[classification]
task = "chat-completions"
modality = "text"
size = "smoke"

[runtime]
adapter = "openai_compatible_chat"
base_url = "http://127.0.0.1:1/v1"
base_url_env = "AIRECEIPES_TEST_BASE_URL"
model = "fake-model"
api_key = "test-key"
system_prompt = "Reply briefly."

[parameters]
temperature = 0.0
max_tokens = 16

[dataset]
prompts = ["Return PONG."]

[kpis]
metrics = ["success_rate", "latency_ms_avg", "output_chars_avg"]
'''.strip(),
            encoding="utf-8",
        )

        result = run_recipe(
            "inference.local-chat-smoke",
            recipes_dir=recipes_dir,
            output_dir=tmp_path / "runs",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["adapter"] == "openai_compatible_chat"
    assert metrics["success_count"] == 1
    assert metrics["error_count"] == 0
    assert metrics["results"][0]["output"] == "fake:Return PONG."
    assert metrics["results"][0]["usage"]["total_tokens"] == 5

    request = _OpenAICompatibleHandler.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["payload"]["model"] == "fake-model"
    assert request["payload"]["stream"] is False
    assert request["payload"]["messages"] == [
        {"role": "system", "content": "Reply briefly."},
        {"role": "user", "content": "Return PONG."},
    ]
    assert request["payload"]["temperature"] == 0.0
    assert request["payload"]["max_tokens"] == 16


class _FailingOpenAICompatibleHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.dumps({"error": {"message": "model is still loading", "code": "not_loaded"}}).encode("utf-8")
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        return None


def test_openai_compatible_errors_include_response_body(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _FailingOpenAICompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        recipes_dir = tmp_path / "recipes"
        recipe_dir = recipes_dir / "inference"
        recipe_dir.mkdir(parents=True)
        (recipe_dir / "local-chat-fail.toml").write_text(
            f'''
id = "inference.local-chat-fail"
category = "inference"
name = "Local OpenAI-compatible chat failure"
description = "Calls an endpoint returning a detailed 400."
tags = ["smoke", "text", "chat", "openai-compatible"]

[classification]
task = "chat-completions"

[runtime]
adapter = "openai_compatible_chat"
base_url = "http://127.0.0.1:{server.server_port}/v1"
model = "fake-model"

[parameters]
max_tokens = 16

[dataset]
prompts = ["Return PONG."]

[kpis]
metrics = ["success_rate"]
'''.strip(),
            encoding="utf-8",
        )

        result = run_recipe("inference.local-chat-fail", recipes_dir=recipes_dir, output_dir=tmp_path / "runs")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["success_count"] == 0
    assert metrics["error_count"] == 1
    assert "HTTP Error 400" in metrics["results"][0]["error"]
    assert "model is still loading" in metrics["results"][0]["error"]
    assert "not_loaded" in metrics["results"][0]["error"]


class _StreamingOpenAICompatibleHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append({"path": self.path, "payload": payload})

        prompt = payload["messages"][-1]["content"]
        if "Translate" in prompt:
            content = "Le chat dort sous la table bleue."
            completion_tokens = 8
        else:
            content = "1. Inspect candidate files.\n2. Review the planned deletions.\n3. Report the safe cleanup command.\nANSWER: ready"
            completion_tokens = 20

        chunks = [
            {"model": payload["model"], "choices": [{"delta": {"content": content[: len(content) // 2]}}]},
            {"model": payload["model"], "choices": [{"delta": {"content": content[len(content) // 2 :]}}]},
            {
                "model": payload["model"],
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": completion_tokens, "total_tokens": completion_tokens + 11},
            },
        ]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        return None


def test_streaming_cases_collect_ttft_tokens_accuracy_and_verticals(tmp_path):
    _StreamingOpenAICompatibleHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _StreamingOpenAICompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        recipes_dir = tmp_path / "recipes"
        recipe_dir = recipes_dir / "inference"
        recipe_dir.mkdir(parents=True)
        (recipe_dir / "vertical-bakeoff.toml").write_text(
            f'''
id = "inference.vertical-bakeoff"
category = "inference"
name = "Vertical bakeoff"
description = "Exercises streaming metrics and case scoring."
tags = ["stream", "kpi"]

[classification]
task = "chat-completions"

[runtime]
adapter = "openai_compatible_chat"
base_url = "http://127.0.0.1:{server.server_port}/v1"
model = "fake-stream-model"

[parameters]
stream = true
stream_include_usage = true
max_tokens = 64

[[dataset.cases]]
id = "translation"
vertical = "translation"
prompt = "Translate this sentence to French."
expected_contains = ["chat", "dort", "bleue"]

[[dataset.cases]]
id = "workflow"
vertical = "agentic"
prompt = "Return a three-step cleanup workflow and final readiness line."
expected_regex = ['(?m)^1\\.', '(?m)^2\\.', '(?m)^3\\.', '(?m)^ANSWER:\\s*ready\\s*$']

[kpis]
metrics = ["accuracy", "workflow_success_rate", "ttft_ms_avg", "tokens_per_sec", "effective_tokens_per_sec"]
'''.strip(),
            encoding="utf-8",
        )

        result = run_recipe("inference.vertical-bakeoff", recipes_dir=recipes_dir, output_dir=tmp_path / "runs")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["success_count"] == 2
    assert metrics["error_count"] == 0
    assert metrics["kpis"]["accuracy"] == 1.0
    assert metrics["kpis"]["workflow_success_rate"] == 1.0
    assert metrics["kpis"]["ttft_ms_avg"] is not None
    assert metrics["kpis"]["tokens_per_sec"] > 0
    assert metrics["kpis"]["effective_tokens_per_sec"] > 0
    assert metrics["verticals"]["translation"]["accuracy"] == 1.0
    assert metrics["verticals"]["agentic"]["case_success_rate"] == 1.0
    assert metrics["results"][0]["completion_tokens"] == 8
    assert metrics["results"][0]["token_count_source"] == "usage"

    request = _StreamingOpenAICompatibleHandler.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["payload"]["stream"] is True
    assert request["payload"]["stream_options"] == {"include_usage": True}
