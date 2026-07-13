import asyncio
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx


EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import post_audit_ab


STYLES = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)


def sample_captions() -> dict[str, str]:
    return {
        "formal": "A person prepares vegetables at a kitchen counter.",
        "sarcastic": "The vegetables bravely accept their extremely dramatic fate.",
        "humorous_tech": "Dinner receives a careful produce patch at the counter.",
        "humorous_non_tech": "The vegetables are getting the full dinner makeover.",
    }


class FakeResponse:
    def __init__(self, content: str, status_code: int = 200):
        self.status_code = status_code
        self.text = content
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class SequencedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class PromptContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_enforces_minimal_factual_edits_and_one_four_caption_request(self):
        captions = sample_captions()
        corrected = dict(captions)
        corrected["formal"] = "A person prepares vegetables at a counter."
        client = SequencedClient([FakeResponse(json.dumps(corrected))])
        stdout, stderr = io.StringIO(), io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            outcome, metadata = await post_audit_ab.audit_caption_set(
                client,
                [f"frame-{i}" for i in range(8)],
                captions,
                api_key="never-print-this-secret",
            )

        prompt = post_audit_ab.AUDIT_PROMPT.lower()
        for required in (
            "minimal edit",
            "unsupported concrete claims",
            "directions",
            "counts",
            "colors",
            "ocr",
            "identities",
            "intentions",
            "backstory",
            "preserve supported details",
            "length band",
            "style",
            "jokes as framing",
            "add no new concrete fact",
            "exact four-key json",
        ):
            self.assertIn(required, prompt)

        self.assertEqual(outcome, corrected)
        self.assertEqual(metadata["status"], "corrected")
        self.assertEqual(metadata["attempts"], 1)
        self.assertEqual(len(client.calls), 1)
        _, request = client.calls[0]
        self.assertEqual(request["json"]["model"], "openai/gpt-5.5")
        self.assertEqual(request["json"]["temperature"], 0.0)
        content = request["json"]["messages"][1]["content"]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 8)
        request_text = "\n".join(item["text"] for item in content if item["type"] == "text")
        for style, caption in captions.items():
            self.assertIn(style, request_text)
            self.assertIn(caption, request_text)
        self.assertNotIn("never-print-this-secret", stdout.getvalue())
        self.assertNotIn("never-print-this-secret", stderr.getvalue())


class ExactKeyParsingTests(unittest.TestCase):
    def test_parser_accepts_only_exact_nonempty_four_style_json(self):
        captions = sample_captions()
        self.assertEqual(post_audit_ab.parse_audit_response(json.dumps(captions)), captions)

        invalid_payloads = (
            {key: value for key, value in captions.items() if key != "formal"},
            {**captions, "metadata": "not allowed"},
            {**captions, "sarcastic": "   "},
            {**captions, "humorous_tech": 42},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    post_audit_ab.parse_audit_response(json.dumps(payload))

        with self.assertRaises(ValueError):
            post_audit_ab.parse_audit_response("```json\n" + json.dumps(captions) + "\n```")


class RetryFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_and_parse_failures_retry_three_times_then_preserve_originals(self):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        captions = sample_captions()
        client = SequencedClient(
            [
                httpx.ConnectError("offline", request=request),
                FakeResponse("not json"),
                FakeResponse('{"formal":"only one key"}'),
            ]
        )

        with patch.object(post_audit_ab.asyncio, "sleep", return_value=None):
            outcome, metadata = await post_audit_ab.audit_caption_set(
                client,
                ["frame"] * 8,
                captions,
                api_key="secret",
            )

        self.assertEqual(len(client.calls), 3)
        self.assertEqual(outcome, captions)
        self.assertIsNot(outcome, captions)
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["attempts"], 3)
        self.assertTrue(metadata["error"])


class LocalFrameTests(unittest.TestCase):
    def test_local_video_path_is_forwarded_to_frame_judge_for_exactly_eight_frames(self):
        calls = []

        def fake_frames_b64(source, n):
            calls.append((source, n))
            self.assertTrue(Path(source).is_file())
            return ["jpeg"] * n

        fake_module = SimpleNamespace(frames_b64=fake_frames_b64)
        with tempfile.TemporaryDirectory() as td:
            video = Path(td, "local clip.mp4")
            video.write_bytes(b"video")
            with patch.dict(sys.modules, {"frame_judge": fake_module}):
                frames = post_audit_ab.load_real_frames(str(video))

        self.assertEqual(frames, ["jpeg"] * 8)
        self.assertEqual(calls, [(str(video), 8)])


class ResultProcessingTests(unittest.IsolatedAsyncioTestCase):
    async def test_frame_failure_is_recorded_in_sidecar_and_preserves_row(self):
        captions = sample_captions()
        results = [{"task_id": "v1", "captions": captions}]
        tasks = {"v1": "missing-local.mp4"}

        def failing_loader(source):
            raise FileNotFoundError(source)

        corrected, sidecar = await post_audit_ab.audit_results(
            results,
            tasks,
            client=object(),
            api_key="secret",
            frame_loader=failing_loader,
        )

        self.assertEqual(corrected, results)
        self.assertIsNot(corrected[0]["captions"], captions)
        self.assertEqual(sidecar["model"], "openai/gpt-5.5")
        self.assertEqual(sidecar["tasks"][0]["task_id"], "v1")
        self.assertEqual(sidecar["tasks"][0]["status"], "failed")
        self.assertEqual(sidecar["tasks"][0]["attempts"], 0)


if __name__ == "__main__":
    unittest.main()
