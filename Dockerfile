FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY recipes ./recipes

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["aireceipes"]
CMD ["run", "inference.echo-smoke", "--output-dir", "runs"]
