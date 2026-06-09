FROM python:3.11-slim AS builder

ENV UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/install/pkg

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /install

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/pkg/lib/python3.11/site-packages

WORKDIR /app

COPY --from=builder /install/pkg /app/pkg
COPY app.py .
COPY templates/ templates/
COPY static/ static/

RUN mkdir -p /app/weights

EXPOSE 5000

CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--worker-tmp-dir", "/dev/shm", "app:app"]
