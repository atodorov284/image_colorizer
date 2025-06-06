# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# copy the uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app


COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project


COPY ./src ./src
COPY ./resnet/best_model ./resnet/best_model

RUN uv sync --frozen

# put the venv on PATH instead of “activating” it
ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /app/src

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
