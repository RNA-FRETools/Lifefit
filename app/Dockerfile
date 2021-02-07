FROM python:3.9-slim-buster as builder
WORKDIR /app
RUN python -m venv .venv && .venv/bin/pip install --no-cache-dir -U pip
COPY requirements.txt .
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim-buster
WORKDIR /app
COPY --from=builder /app /app
COPY app.py .
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
CMD streamlit run --server.port $PORT app.py
