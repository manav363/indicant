# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Contracts first: it changes least often, so this layer stays cached across
# ordinary service edits.
COPY packages/contracts /app/packages/contracts
RUN pip install --no-cache-dir /app/packages/contracts

COPY services/worker /app/services/worker
RUN pip install --no-cache-dir /app/services/worker

# The lake is a mounted volume, never baked into the image.
ENV INDICANT_LAKE_ROOT=/data

EXPOSE 8000
CMD ["uvicorn", "worker.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
