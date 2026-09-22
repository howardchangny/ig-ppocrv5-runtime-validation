import base64
import hashlib
import json
import os
import platform
import time


def emit(tag, value):
    print(f"{tag}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}", flush=True)


parts = []
i = 0
while True:
    key = f"FRAME_B64_{i:03d}"
    if key not in os.environ:
        break
    parts.append(os.environ[key])
    i += 1

frame_bytes = base64.b64decode("".join(parts), validate=True)
frame_path = "/tmp/ppocr_test_frame_30s.png"
with open(frame_path, "wb") as f:
    f.write(frame_bytes)

emit("RUNTIME", {"python": platform.python_version(), "platform": platform.platform()})
emit("INPUT", {
    "path": frame_path,
    "bytes": len(frame_bytes),
    "sha256": hashlib.sha256(frame_bytes).hexdigest(),
    "expected_dimensions": [720, 1280],
})

import cv2
import paddle
import paddleocr
from paddleocr import PaddleOCR

image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError("OpenCV could not decode the test frame")
h, w = image.shape[:2]
emit("DECODED_IMAGE", {"width": w, "height": h, "channels": image.shape[2]})
emit("PACKAGES", {"paddlepaddle": paddle.__version__, "paddleocr": paddleocr.__version__})

model_config = {
    "text_detection_model_name": "PP-OCRv5_mobile_det",
    "text_recognition_model_name": "PP-OCRv5_mobile_rec",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "device": "cpu",
}
emit("MODEL_CONFIG", model_config)

t0 = time.perf_counter()
ocr = PaddleOCR(**model_config)
initialization_seconds = time.perf_counter() - t0

t1 = time.perf_counter()
predictions = list(ocr.predict(input=image))
inference_seconds = time.perf_counter() - t1
if len(predictions) != 1:
    raise RuntimeError(f"Expected exactly one prediction, got {len(predictions)}")

result = predictions[0]
payload = getattr(result, "json", None)
if callable(payload):
    payload = payload()
if isinstance(payload, str):
    payload = json.loads(payload)
if not isinstance(payload, dict):
    raise RuntimeError(f"Unsupported PaddleOCR result JSON type: {type(payload)!r}")
data = payload.get("res", payload)
texts = [str(x) for x in data.get("rec_texts", [])]
scores = [float(x) for x in data.get("rec_scores", [])]
polys = data.get("rec_polys", [])

if len(texts) != len(scores):
    raise RuntimeError(f"Text/score count mismatch: {len(texts)} vs {len(scores)}")

raw_text = "\n".join(texts)
avg_confidence = sum(scores) / len(scores) if scores else 0.0
emit("TIMING", {
    "initialization_seconds": initialization_seconds,
    "inference_seconds": inference_seconds,
})
emit("SUMMARY", {
    "line_count": len(texts),
    "average_confidence": avg_confidence,
    "detected_polygon_count": len(polys),
})
emit("RAW_LINES", [
    {"index": n + 1, "text": text, "confidence": scores[n]}
    for n, text in enumerate(texts)
])
print("RAW_OCR_BEGIN", flush=True)
print(raw_text, flush=True)
print("RAW_OCR_END", flush=True)
print("RUNTIME_VALIDATION_COMPLETE", flush=True)
