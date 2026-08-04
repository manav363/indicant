# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LightGBM links against the OpenMP runtime, which python:*-slim does not carry.
# Without it the import fails at MODEL LOAD time, not at build time, so the
# image builds clean and then crash-loops on startup with
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Contracts first: it changes least often, so this layer stays cached across
# ordinary service edits.
COPY packages/contracts /app/packages/contracts
RUN pip install --no-cache-dir /app/packages/contracts

COPY services/intelligence /app/services/intelligence
RUN pip install --no-cache-dir /app/services/intelligence

# The lake is a mounted volume, never baked into the image.
ENV INDICANT_LAKE_ROOT=/data

EXPOSE 8000
CMD ["uvicorn", "intelligence.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
