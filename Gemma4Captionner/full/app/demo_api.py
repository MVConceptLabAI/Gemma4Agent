"""Authenticated asynchronous API for the public Gemma 4 demonstration.

This process is designed to run on the GPU instance, behind the Cloudflare
Worker in ``demo-worker``. It deliberately exposes no Jupyter functionality.
"""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import os
import socket
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, request

from app.ensemble import caption_ensemble, caption_ensemble_file
from app.models import REQUIRED_STYLES, normalize_captions

Status = Literal["queued", "running", "complete", "failed"]
_MAX_JOBS = 100
_JOB_RETENTION_S = 2 * 60 * 60
_token = os.environ.get("DEMO_ORIGIN_TOKEN", "")

if not _token:
    raise RuntimeError("DEMO_ORIGIN_TOKEN must be set before starting the public demo API")

app = Flask(__name__)
_UPLOAD_MAX_BYTES = int(os.environ.get("DEMO_UPLOAD_MAX_BYTES", str(80 * 1024 * 1024)))
_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
app.config["MAX_CONTENT_LENGTH"] = _UPLOAD_MAX_BYTES
_jobs_lock = threading.Lock()
_jobs: dict[str, "CaptionJob"] = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemma4-caption")


@dataclass
class CaptionJob:
    created_at: float
    status: Status = "queued"
    captions: dict[str, str] | None = None
    upload_dir: str | None = None


def _authorized() -> bool:
    supplied = request.headers.get("X-Demo-Origin-Token", "")
    return bool(supplied) and hmac.compare_digest(supplied, _token)


def _public_https_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("video_url must be a string")
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("provide a public HTTPS video URL")
    if parsed.port not in {None, 443}:
        raise ValueError("only the standard HTTPS port is allowed")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("video host could not be resolved") from exc
    for _, _, _, _, sockaddr in addresses:
        if not ipaddress.ip_address(sockaddr[0]).is_global:
            raise ValueError("private and local video hosts are not allowed")
    return url


def _trim_old_jobs(now: float) -> None:
    expired = [job_id for job_id, job in _jobs.items() if now - job.created_at > _JOB_RETENTION_S]
    for job_id in expired:
        _cleanup_job(_jobs.pop(job_id, None))
    while len(_jobs) >= _MAX_JOBS:
        oldest = min(_jobs, key=lambda key: _jobs[key].created_at)
        _cleanup_job(_jobs.pop(oldest, None))


def _cleanup_job(job: CaptionJob | None) -> None:
    if job and job.upload_dir:
        shutil.rmtree(job.upload_dir, ignore_errors=True)
        job.upload_dir = None


def _run_caption(job_id: str, video_url: str | None = None, video_path: str | None = None) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.status = "running"
    try:
        if video_path:
            captions = asyncio.run(caption_ensemble_file(Path(video_path), list(REQUIRED_STYLES)))
        else:
            captions = asyncio.run(caption_ensemble(video_url or "", list(REQUIRED_STYLES)))
        result = normalize_captions(captions, list(REQUIRED_STYLES))
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.captions = result
                job.status = "complete"
    except Exception:  # The detailed stack trace stays in the private GPU log.
        app.logger.exception("caption job %s failed", job_id)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.status = "failed"
    finally:
        with _jobs_lock:
            _cleanup_job(_jobs.get(job_id))


@app.before_request
def require_worker_token() -> Response | None:
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    return None


@app.get("/health")
def health() -> Response:
    return jsonify({"status": "ok", "queue_limit": 1})


@app.post("/jobs")
def create_job() -> Response:
    try:
        payload = request.get_json(force=True)
        video_url = _public_https_url(payload.get("video_url"))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = os.urandom(18).hex()
    now = time.time()
    with _jobs_lock:
        _trim_old_jobs(now)
        _jobs[job_id] = CaptionJob(created_at=now)
    _executor.submit(_run_caption, job_id, video_url)
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.post("/uploads")
def create_upload_job() -> Response:
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify({"error": "choose a video file"}), 400
    extension = Path(video.filename).suffix.lower()
    if extension not in _ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": "use an MP4, MOV, WebM, or MKV video"}), 400

    upload_dir = Path(tempfile.mkdtemp(prefix="gemma4-demo-"))
    video_path = upload_dir / f"upload{extension}"
    try:
        video.save(video_path)
        if video_path.stat().st_size == 0:
            raise ValueError("uploaded video is empty")
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(upload_dir, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400

    job_id = os.urandom(18).hex()
    now = time.time()
    with _jobs_lock:
        _trim_old_jobs(now)
        _jobs[job_id] = CaptionJob(created_at=now, upload_dir=str(upload_dir))
    _executor.submit(_run_caption, job_id, None, str(video_path))
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.errorhandler(413)
def uploaded_file_too_large(_: object) -> Response:
    return jsonify({"error": f"video exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MB demo limit"}), 413


@app.get("/jobs/<job_id>")
def read_job(job_id: str) -> Response:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        response: dict[str, object] = {"job_id": job_id, "status": job.status}
        if job.status == "complete":
            response["captions"] = job.captions
        elif job.status == "failed":
            response["error"] = "captioning failed; see the private GPU service logs"
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("DEMO_PORT", "8799")), threaded=True)
