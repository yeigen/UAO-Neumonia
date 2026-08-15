FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends tk libgl1 libglib2.0-0 scrot && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

COPY main.py ./
COPY src/ src/
COPY test/ test/
COPY scripts/ scripts/

CMD ["uv", "run", "--no-sync", "main.py"]
