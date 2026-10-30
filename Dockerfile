FROM python:3.11-slim

# Install uv (the package manager you already use)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 1. Dependencies first (cached unless pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Code, model and data
COPY src/ ./src/
COPY models/final/ ./models/final/
COPY data/processed/laliga.csv ./data/processed/laliga.csv
COPY data/features/laliga_features.csv ./data/features/laliga_features.csv

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]