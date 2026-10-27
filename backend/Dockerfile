# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy CMAKE_ARGS="-DGGML_NATIVE=OFF"
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra all --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra all

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home voicedesk
WORKDIR /app
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/hf \
    VOICEDESK_LLAMACPP_MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    VOICEDESK_METRICS_PATH=/data/metrics.sqlite
RUN mkdir -p /cache/hf /data /models && chown -R voicedesk /cache /data
USER voicedesk
EXPOSE 8000
HEALTHCHECK --interval=15s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["python", "-m", "voicedesk"]
