FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_LINK_MODE=copy

COPY .git .git
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

COPY reg_transfo/ reg_transfo/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --python 3.11

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "reg_transfo/main.py"]
