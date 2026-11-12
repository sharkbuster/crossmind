FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XM_HOST=0.0.0.0 \
    XM_PORT=8000

WORKDIR /app

RUN addgroup --system crossmind && adduser --system --ingroup crossmind crossmind

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY apps ./apps
COPY data ./data
RUN pip install --no-cache-dir --no-deps -e .

RUN mkdir -p var/traces reports && chown -R crossmind:crossmind /app
USER crossmind

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz').status==200 else 1)"

CMD ["uvicorn", "crossmind.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
