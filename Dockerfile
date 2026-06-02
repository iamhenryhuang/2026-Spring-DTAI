FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /install

COPY requirements.txt .
RUN pip install --prefix=/install/pkg -r requirements.txt


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

CMD ["/app/pkg/bin/gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
