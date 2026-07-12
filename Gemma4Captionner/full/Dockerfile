# =============================================================================
# AMD Developer Hackathon: ACT II — Track 2 Video Captioning Agent
# Base slim + FFmpeg. Build for linux/amd64 explicitly (judging VM runs amd64).
#
#   docker buildx build --platform linux/amd64 \
#     --tag ghcr.io/<you>/track2-captioner:latest --push .
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
# Submission engine = ENSEMBLE (measured best on the official jury distribution:
# 0.942 accuracy, ~14.8 correct details/caption, lowest contradictions). Four
# frontier vision models observe the frames; Claude Opus 4.5 cross-references and
# writes. To fall back to the single-model pipeline, set CAPTION_ENGINE=pipeline.
ENV PYTHONPATH=/app \
    CAPTION_ENGINE=ensemble \
    ENSEMBLE_OBSERVERS=openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.5,qwen/qwen3-vl-235b-a22b-instruct \
    ENSEMBLE_WRITER=google/gemma-4-31b-it \
    STYLE_EXEMPLARS=1 \
    STRICT_GROUNDING=0 \
    WRITER_TEMP=0.5 \
    W4_STYLE_SPLIT=1 \
    TIMESTAMP_FRAMES=0 \
    MAX_CAPTION_CHARS=1600 \
    OPENROUTER_VLM_MODEL=qwen/qwen3-vl-235b-a22b-instruct \
    OPENROUTER_STYLE_MODEL=google/gemma-4-31b-it \
    PROVIDER_ORDER=openrouter,groq,fireworks \
    STYLE_PROVIDER_ORDER=openrouter,fireworks,groq \
    STYLE_MODEL=accounts/fireworks/models/gpt-oss-120b \
    STYLE_REASONING_EFFORT=low \
    STYLE_MAX_TOKENS=1400 \
    DETERMINISTIC_FORMAL=1 \
    NUM_FRAMES=10 \
    FRAME_MAX_EDGE=896 \
    GROQ_MAX_IMAGES=4 \
    GROQ_FRAME_MAX_EDGE=448 \
    HTTP_429_RETRIES=5 \
    HTTP_429_MAX_WAIT_S=45 \
    RETRY_AFTER_GIVEUP_S=60 \
    DESCRIBE_MAX_TOKENS=1300 \
    SCENE_DETECT_ENABLED=0 \
    MAX_CONCURRENCY=2 \
    PER_TASK_TIMEOUT_S=150 \
    GLOBAL_BUDGET_S=540
# Judging VM = 2 vCPU / 4 GB RAM (Participant Guide p.5). Scene-detection
# decodes the ENTIRE UHD clip per video and three parallel ffmpeg passes
# thrash 2 cores into the 10-minute wall - uniform -ss seeks are near-free.

# API keys arrive as build args at publish time only (CI secrets) — the repo
# and default builds stay key-free; without keys the image degrades safely.
ARG OPENROUTER_API_KEY=""
ARG GROQ_API_KEY=""
ARG FIREWORKS_API_KEY=""
ENV OPENROUTER_API_KEY=${OPENROUTER_API_KEY} \
    GROQ_API_KEY=${GROQ_API_KEY} \
    FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

# Sanity: must start and be ready in < 60s. Keep the container thin.
CMD ["python", "-u", "-m", "app.main"]
