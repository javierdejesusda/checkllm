# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y make git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.lock /app/
RUN pip install -r requirements.lock \
    && pip install pytest

COPY . /app
RUN pip install --no-deps -e .

ENTRYPOINT ["make"]
CMD ["reproduce"]
