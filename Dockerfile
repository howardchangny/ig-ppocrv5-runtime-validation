FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PADDLE_PDX_MODEL_SOURCE=BOS \
    PADDLE_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_use_mkldnn=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    paddlepaddle==3.2.1 \
    paddleocr==3.5.0

WORKDIR /app
COPY railway_ppocrv5_runtime.py /app/railway_ppocrv5_runtime.py

CMD ["python", "/app/railway_ppocrv5_runtime.py"]
