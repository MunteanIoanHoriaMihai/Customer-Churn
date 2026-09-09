# ---- builder ----
FROM python:3.13-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.in-project true

COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root

COPY README.md ./
COPY src/ ./src/
RUN poetry install --without dev

# ---- runtime ----
FROM python:3.13-slim

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY models/ ./models/
COPY mlflow.db ./
COPY mlruns/ ./mlruns/

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8000

CMD ["uvicorn", "customer_churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
