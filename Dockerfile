FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY recipes ./recipes
COPY skills ./skills
COPY mcp ./mcp

RUN pip install --no-cache-dir -e .

# Competition default: read /input/tasks.json and write /output/results.json.
# Optional off-contest tooling remains available with, for example:
#   docker run --rm --entrypoint aireceipes <image> list
ENTRYPOINT ["python", "-m", "aireceipes.track1_agent"]
