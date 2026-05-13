# syntax=docker/dockerfile:1.7

# --- Build stage -----------------------------------------------------------
FROM python:3.12-slim AS build
WORKDIR /app

# Install uv (single static binary)
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /usr/local/bin/

# Cache deps separately from source — only re-resolves when lock changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# Install the project itself
COPY src src
RUN uv sync --frozen --no-dev

# --- Runtime stage ---------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src

RUN chown -R app:app /app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["sh", "-c", "python -m knowledge_bot.ingest && exec uvicorn knowledge_bot.main:app --host 0.0.0.0 --port 8000"]
