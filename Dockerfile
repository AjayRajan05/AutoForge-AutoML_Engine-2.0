FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY autoforge ./autoforge
COPY api ./api
COPY benchmarking ./benchmarking
COPY cli ./cli
COPY config ./config
COPY core ./core
COPY ensemble ./ensemble
COPY execution ./execution
COPY features ./features
COPY input_output ./input_output
COPY intelligence ./intelligence
COPY models ./models
COPY optimizer ./optimizer
COPY persistence ./persistence
COPY processors ./processors
COPY registry ./registry
COPY serving ./serving
COPY systemization ./systemization
COPY utils ./utils
COPY main.py bootstrap.py ./

RUN pip install --no-cache-dir -e ".[serve]"

ENV AUTOFORGE_HOST=0.0.0.0
ENV AUTOFORGE_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "serving.run", "--host", "0.0.0.0", "--port", "8000"]
