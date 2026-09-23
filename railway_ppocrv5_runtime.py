import base64
import binascii
import gc
import hashlib
import json
import os
import platform
import re
import threading
import time
import unicodedata
from difflib import SequenceMatcher

import cv2
import numpy as np
import paddle
import paddleocr
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field


MODEL_CONFIG = {
    "text_detection_model_name": "PP-OCRv5_mobile_det",
    "text_recognition_model_name": "PP-OCRv5_mobile_rec",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "device": "cpu",
}
MAX_FRAMES = 7
MAX_FRAME_BYTES = 2_500_000
MAX_TOTAL_BYTES = 12_000_000
API_TOKEN = os.environ.get("OCR_API_TOKEN", "")
if len(API_TOKEN) < 32:
    raise RuntimeError("OCR_API_TOKEN must be configured with at least 32 characters")


def emit(tag, value):
    print(
        f"{tag}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}",
        flush=True,
    )


startup_started = time.perf_counter()
ocr = PaddleOCR(**MODEL_CONFIG)
initialization_seconds = time.perf_counter() - startup_started
ocr_lock = threading.Lock()
emit(
    "SERVICE_READY",
    {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "paddlepaddle": paddle.__version__,
        "paddleocr": paddleocr.__version__,
        "model_config": MODEL_CONFIG,
        "initialization_seconds": initialization_seconds,
        "token_required": True,
    },
)


class FrameInput(BaseModel):
    time: float | None = None
    dataUrl: str = Field(min_length=32)


class OcrRequest(BaseModel):
    frames: list[FrameInput] = Field(min_length=1, max_length=MAX_FRAMES)


def decode_frame(data_url: str) -> tuple[bytes, np.ndarray]:
    raw = data_url.split(",", 1)[1] if data_url.startswith("data:") and "," in data_url else data_url
    try:
        frame_bytes = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid base64 frame") from error
    if not frame_bytes or len(frame_bytes) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Frame size is outside the allowed range")
    image = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="OpenCV could not decode a frame")
    height, width = image.shape[:2]
    if width < 32 or height < 32 or width > 2400 or height > 4000:
        raise HTTPException(status_code=400, detail="Frame dimensions are outside the allowed range")
    return frame_bytes, image


def result_data(result) -> dict:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported PaddleOCR result JSON type: {type(payload)!r}")
    return payload.get("res", payload)


def canonical(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(ch for ch in value if ch.isalnum() or "\u3400" <= ch <= "\u9fff")


def same_line(a: str, b: str) -> bool:
    left, right = canonical(a), canonical(b)
    if not left or not right:
        return False
    if left == right:
        return True
    shortest, longest = sorted((left, right), key=len)
    if len(shortest) >= 6 and shortest in longest and len(shortest) / len(longest) >= 0.88:
        return True
    return min(len(left), len(right)) >= 6 and SequenceMatcher(None, left, right).ratio() >= 0.92


def merge_line(rows: list[dict], candidate: dict) -> None:
    for index, existing in enumerate(rows):
        if same_line(existing["text"], candidate["text"]):
            existing["frames"] = sorted(set(existing["frames"] + candidate["frames"]))
            if candidate["confidence"] > existing["confidence"]:
                candidate["frames"] = existing["frames"]
                rows[index] = candidate
            return
    rows.append(candidate)


app = FastAPI(title="IG Archive PP-OCRv5", version="1.5.8.001")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-IGAD-Token"],
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "1.5.8.001",
        "models": ["PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"],
        "initializationSeconds": initialization_seconds,
    }


@app.post("/v1/ocr")
def recognize(request: OcrRequest, x_igad_token: str | None = Header(default=None)):
    if x_igad_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid OCR token")
    started = time.perf_counter()
    merged: list[dict] = []
    frame_summaries = []
    total_bytes = 0
    with ocr_lock:
        for frame_index, frame in enumerate(request.frames):
            frame_bytes, image = decode_frame(frame.dataUrl)
            total_bytes += len(frame_bytes)
            if total_bytes > MAX_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="Combined frame size exceeds the limit")
            inference_started = time.perf_counter()
            predictions = list(ocr.predict(input=image))
            inference_seconds = time.perf_counter() - inference_started
            if len(predictions) != 1:
                raise RuntimeError(f"Expected one prediction, got {len(predictions)}")
            data = result_data(predictions[0])
            texts = [str(value).strip() for value in data.get("rec_texts", [])]
            scores = [float(value) for value in data.get("rec_scores", [])]
            if len(texts) != len(scores):
                raise RuntimeError("Text and score counts do not match")
            kept = 0
            for text, score in zip(texts, scores):
                if not text:
                    continue
                kept += 1
                merge_line(
                    merged,
                    {"text": text, "confidence": score, "frames": [frame_index]},
                )
            height, width = image.shape[:2]
            frame_summaries.append(
                {
                    "index": frame_index,
                    "time": frame.time,
                    "width": width,
                    "height": height,
                    "bytes": len(frame_bytes),
                    "sha256": hashlib.sha256(frame_bytes).hexdigest(),
                    "lineCount": kept,
                    "inferenceSeconds": inference_seconds,
                }
            )
            del predictions, data, texts, scores, image, frame_bytes
            gc.collect()

    confidences = [row["confidence"] for row in merged]
    raw_text = [row["text"] for row in merged]
    mentions = []
    for line in raw_text:
        for match in re.findall(r"@[A-Za-z0-9._]{1,30}", line):
            handle = match[1:]
            if handle.casefold() not in [value.casefold() for value in mentions]:
                mentions.append(handle)
    response = {
        "ok": True,
        "engine": "Railway official PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec",
        "modelConfig": MODEL_CONFIG,
        "sampledFrames": len(request.frames),
        "sampledTimes": [frame.time for frame in request.frames if frame.time is not None],
        "rawText": raw_text,
        "lines": merged,
        "mentions": mentions,
        "lineCount": len(raw_text),
        "averageConfidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "initializationSeconds": initialization_seconds,
        "inferenceSeconds": sum(row["inferenceSeconds"] for row in frame_summaries),
        "requestSeconds": time.perf_counter() - started,
        "frames": frame_summaries,
    }
    emit(
        "OCR_REQUEST_COMPLETE",
        {
            "sampled_frames": response["sampledFrames"],
            "line_count": response["lineCount"],
            "average_confidence": response["averageConfidence"],
            "request_seconds": response["requestSeconds"],
        },
    )
    return response

