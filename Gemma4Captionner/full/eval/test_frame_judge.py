import asyncio
import importlib.util
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx


def _load_frame_judge():
    module_path = Path(__file__).with_name("frame_judge.py")
    spec = importlib.util.spec_from_file_location("frame_judge_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with tempfile.TemporaryDirectory() as td:
        previous_cwd = os.getcwd()
        os.chdir(td)
        try:
            with patch.object(asyncio, "run", side_effect=lambda coroutine: coroutine.close()):
                spec.loader.exec_module(module)
        finally:
            os.chdir(previous_cwd)
    return module


frame_judge = _load_frame_judge()


class FramesB64Tests(unittest.TestCase):
    def _exercise_local_source(self, source: str) -> list[str]:
        def fake_run(command, check):
            self.assertNotEqual(command[0], "curl", "local videos must not be downloaded")
            self.assertEqual(command[0], "ffmpeg")
            Path(command[-1]).write_bytes(b"jpeg")
            return subprocess.CompletedProcess(command, 0)

        def fake_check_output(command):
            self.assertEqual(command[0], "ffprobe")
            return b"1.0"

        with patch.object(frame_judge.subprocess, "run", side_effect=fake_run), patch.object(
            frame_judge.subprocess, "check_output", side_effect=fake_check_output
        ):
            return frame_judge.frames_b64(source, n=1)

    def test_existing_path_and_file_url_bypass_curl(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td, "cached clip.mp4")
            video.write_bytes(b"video")
            for source in (str(video), video.as_uri()):
                with self.subTest(source=source):
                    self.assertEqual(self._exercise_local_source(source), ["anBlZw=="])

    def test_remote_download_retries_curl_exit_28(self):
        curl_attempts = 0

        def fake_run(command, check):
            nonlocal curl_attempts
            if command[0] == "curl":
                curl_attempts += 1
                if curl_attempts < 3:
                    raise subprocess.CalledProcessError(28, command)
                Path(command[-1]).write_bytes(b"video")
            else:
                Path(command[-1]).write_bytes(b"jpeg")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(frame_judge.subprocess, "run", side_effect=fake_run), patch.object(
            frame_judge.subprocess, "check_output", return_value=b"1.0"
        ):
            self.assertEqual(frame_judge.frames_b64("https://example.test/clip.mp4", n=1), ["anBlZw=="])

        self.assertEqual(curl_attempts, 3)


class JudgeRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_error_is_retried_instead_of_crashing(self):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

        class Response:
            status_code = 200
            text = '{"choices":[{"message":{"content":"ok"}}]}'

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"accuracy": 0.8, "style": 0.7, "wrong_claims": []}'
                            }
                        }
                    ]
                }

        class Client:
            def __init__(self):
                self.attempts = 0

            async def post(self, *args, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise httpx.ConnectError("offline", request=request)
                return Response()

        client = Client()
        with patch.dict(frame_judge.os.environ, {"OPENROUTER_API_KEY": "test-key"}), patch.object(
            frame_judge.asyncio, "sleep", new=AsyncMock()
        ):
            score = await frame_judge.judge_one(client, ["frame"], "formal", "A factual caption.")

        self.assertEqual(client.attempts, 2)
        self.assertEqual(score["accuracy"], 0.8)


class EvaluationAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_clips_and_nonfinite_judges_are_marked_and_excluded(self):
        results = [
            {"task_id": "bad-video", "captions": {"formal": "one", "sarcastic": "two"}},
            {
                "task_id": "good-video",
                "captions": {
                    "formal": "three",
                    "sarcastic": "four",
                    "humorous_tech": "five",
                },
            },
        ]
        tasks = {"bad-video": "bad.mp4", "good-video": "good.mp4"}

        def frame_loader(url):
            if url == "bad.mp4":
                raise subprocess.CalledProcessError(28, ["curl"])
            return ["frame"]

        async def judge_fn(client, frames, style, caption):
            if style == "sarcastic":
                raise httpx.ConnectError("offline")
            if style == "humorous_tech":
                return {"accuracy": math.nan, "style": math.nan, "skipped": "unparseable"}
            return {"accuracy": 0.8, "style": 0.6, "wrong_claims": []}

        rows, total_accuracy, total_style, count = await frame_judge.evaluate_results(
            results,
            tasks,
            client=object(),
            frame_loader=frame_loader,
            judge_fn=judge_fn,
        )

        self.assertEqual(len(rows), 5)
        self.assertEqual(count, 1)
        self.assertEqual(total_accuracy, 0.8)
        self.assertEqual(total_style, 0.6)
        self.assertEqual(sum("skipped" in row["judge"] for row in rows), 4)


if __name__ == "__main__":
    unittest.main()
