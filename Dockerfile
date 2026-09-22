FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PADDLE_PDX_MODEL_SOURCE=BOS \
    PADDLE_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_use_mkldnn=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    paddlepaddle==3.2.1 \
    paddleocr==3.5.0 \
    fastapi==0.116.1 \
    uvicorn==0.35.0

RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det',text_recognition_model_name='PP-OCRv5_mobile_rec',use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,device='cpu')"

WORKDIR /app
COPY railway_ppocrv5_runtime.py /app/railway_ppocrv5_runtime.py

CMD ["sh", "-c", "uvicorn railway_ppocrv5_runtime:app --host 0.0.0.0 --port ${PORT:-8080}"]
