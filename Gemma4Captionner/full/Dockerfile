# =============================================================================
# AMD Developer Hackathon: ACT II — Track 2 Video Captioning Agent
# Base slim + FFmpeg. Build for linux/amd64 explicitly (judging VM runs amd64).
#
#   docker buildx build --platform linux/amd64 \
#     --tag ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v20 --push .
# =============================================================================
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# FFmpeg is used for keyframe extraction + audio track split.
# ca-certificates lets requests/openai talk HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# The harness mounts /input and /output at runtime and injects NO env vars,
# so the submission profile is pinned here. Non-secret config is plain ENV.
# V20.5: one bounded six-frame request per video, leaving margin under the
# 30-second general response-time rule. Local repairs keep accepted visual
# anchors; expensive recovery is disabled in this submission profile.
ENV PYTHONPATH=/app \
    CAPTION_ENGINE=gemma_hybrid \
    GEMMA_FAST_MODEL=google/gemma-4-26b-a4b-it \
    GEMMA_FAST_MAX_TOKENS=350 \
    DEMO_GEMMA_MODEL=google/gemma-4-26b-a4b-it \
    DEMO_VIDEO_MODEL=google/gemma-4-26b-a4b-it \
    DEMO_OBSERVATION_CONCURRENCY=9 \
    VIDEO_CONTEXT_MAX_SECONDS=60 \
    VIDEO_CONTEXT_MAX_BYTES=12000000 \
    HYBRID_FAST_TIMEOUT_S=26 \
    HYBRID_RECOVERY_TIMEOUT_S=0 \
    HYBRID_RECOVERY_MAX_PER_RUN=0 \
    HYBRID_BATCH_RECOVERY_TIMEOUT_S=0 \
    HYBRID_BATCH_RECOVERY_MAX=0 \
    SCENE_DETECT_ENABLED=0 \
    TIMESTAMP_FRAMES=0 \
    TRUST_GENERATED_HUMOROUS_TECH=1 \
    MAX_CAPTION_CHARS=500 \
    NUM_FRAMES=6 \
    FRAME_MAX_EDGE=640 \
    MAX_CONCURRENCY=1 \
    PER_TASK_TIMEOUT_S=28 \
    GLOBAL_BUDGET_S=540 \
    GLOBAL_BUDGET_RESERVE_S=30
# Judging VM = 2 vCPU / 4 GB RAM (Participant Guide p.5). Scene-detection
# decodes the ENTIRE UHD clip per video and three parallel ffmpeg passes
# thrash 2 cores into the 10-minute wall - uniform -ss seeks are near-free.

# API keys arrive as build args at publish time only (CI secrets) — the repo
# and default builds stay key-free; without keys the image degrades safely.
ARG OPENROUTER_API_KEY=""
ENV OPENROUTER_API_KEY=${OPENROUTER_API_KEY}

# Sanity: must start and be ready in < 60s. Keep the container thin.
CMD ["python", "-u", "-m", "app.main"]
