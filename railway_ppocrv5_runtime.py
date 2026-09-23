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
from opencc import OpenCC
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
traditionalizer = OpenCC("s2tw")
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


def traditional_text(text: str) -> str:
    return traditionalizer.convert(unicodedata.normalize("NFKC", str(text))).strip()


def text_similarity(a: str, b: str) -> float:
    left, right = canonical(a), canonical(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    shortest, longest = sorted((left, right), key=len)
    if len(shortest) >= 6 and shortest in longest and len(shortest) / len(longest) >= 0.88:
        return len(shortest) / len(longest)
    return SequenceMatcher(None, left, right).ratio()


def normalized_box(poly, width: int, height: int) -> list[float] | None:
    try:
        points = list(poly)
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        if not xs or not ys:
            return None
        return [
            max(0.0, min(xs) / width),
            max(0.0, min(ys) / height),
            min(1.0, max(xs) / width),
            min(1.0, max(ys) / height),
        ]
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def same_position(a: list[float] | None, b: list[float] | None) -> bool:
    if not a or not b:
        return False
    aw, ah = max(a[2] - a[0], 0.001), max(a[3] - a[1], 0.001)
    bw, bh = max(b[2] - b[0], 0.001), max(b[3] - b[1], 0.001)
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    overlap_x = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / min(aw, bw)
    return (
        abs(acy - bcy) <= max(0.02, max(ah, bh) * 1.15)
        and (overlap_x >= 0.30 or abs(acx - bcx) <= max(0.05, max(aw, bw) * 0.35))
        and 0.45 <= ah / bh <= 2.20
    )


def best_observation(observations: list[dict]) -> dict:
    if len(observations) == 1:
        return observations[0]
    best = observations[0]
    best_score = -1.0
    for item in observations:
        similarities = [
            text_similarity(item["text"], other["text"])
            for other in observations
            if other is not item
        ]
        consensus = sum(similarities) / len(similarities) if similarities else 0.0
        score = consensus + 0.12 * item["confidence"] + min(len(canonical(item["text"])), 40) * 0.001
        if score > best_score:
            best, best_score = item, score
    return best


def merge_line(rows: list[dict], candidate: dict) -> None:
    for index, existing in enumerate(rows):
        similarity = text_similarity(existing["text"], candidate["text"])
        different_frame = candidate["frame"] not in existing["frames"]
        positional_match = different_frame and same_position(existing.get("box"), candidate.get("box"))
        same_text = similarity >= 0.92 or (
            min(len(canonical(existing["text"])), len(canonical(candidate["text"]))) >= 6
            and similarity >= 0.82
        )
        positional_variant = positional_match and similarity >= (
            0.52 if max(len(canonical(existing["text"])), len(canonical(candidate["text"]))) >= 12 else 0.62
        )
        if same_text or positional_variant:
            existing["observations"].append(candidate)
            existing["frames"] = sorted(set(existing["frames"] + [candidate["frame"]]))
            representative = best_observation(existing["observations"])
            existing["text"] = representative["text"]
            existing["confidence"] = representative["confidence"]
            existing["box"] = representative.get("box")
            return
    rows.append(
        {
            "text": candidate["text"],
            "confidence": candidate["confidence"],
            "frames": [candidate["frame"]],
            "box": candidate.get("box"),
            "observations": [candidate],
        }
    )


def token_candidates(text: str) -> list[str]:
    return [
        value
        for value in re.findall(r"[@＠]?[A-Za-z0-9._]{4,30}", text)
        if any(character.isalpha() for character in value)
    ]


def mention_candidates(rows: list[dict]) -> list[str]:
    mentions: list[str] = []

    def add(value: str):
        handle = value.lstrip("@＠").strip("._").casefold()
        if re.fullmatch(r"[a-z0-9._]{3,30}", handle) and handle not in mentions:
            mentions.append(handle)

    for row in rows:
        observations = row.get("observations", [])
        tokens = [token for item in observations for token in token_candidates(item["text"])]
        for token in tokens:
            if token.startswith(("@", "＠")):
                add(token)

        dotted = [token.lstrip("@＠") for token in tokens if "." in token]
        if dotted:
            medoid = max(
                dotted,
                key=lambda value: sum(text_similarity(value, other) for other in dotted),
            )
            if len(observations) >= 2 or medoid.startswith(("@", "＠")):
                if len(medoid) >= 6 and medoid[0].isupper() and medoid[1].islower():
                    trimmed = medoid[1:]
                    if any(text_similarity(trimmed, other) > text_similarity(medoid, other) for other in dotted):
                        medoid = trimmed
                add(medoid)

    return mentions


app = FastAPI(title="IG Archive PP-OCRv5", version="1.5.8.002")
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
        "version": "1.5.8.002",
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
            texts = [traditional_text(value) for value in data.get("rec_texts", [])]
            scores = [float(value) for value in data.get("rec_scores", [])]
            polys = data.get("rec_polys", [])
            if len(texts) != len(scores):
                raise RuntimeError("Text and score counts do not match")
            kept = 0
            height, width = image.shape[:2]
            for line_index, (text, score) in enumerate(zip(texts, scores)):
                if not text:
                    continue
                kept += 1
                merge_line(
                    merged,
                    {
                        "text": text,
                        "confidence": score,
                        "frame": frame_index,
                        "box": normalized_box(polys[line_index], width, height)
                        if line_index < len(polys)
                        else None,
                    },
                )
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
            del predictions, data, texts, scores, polys, image, frame_bytes
            gc.collect()

    confidences = [row["confidence"] for row in merged]
    raw_text = [row["text"] for row in merged]
    mentions = mention_candidates(merged)
    response_lines = [
        {
            "text": row["text"],
            "confidence": row["confidence"],
            "frames": row["frames"],
            "box": row.get("box"),
        }
        for row in merged
    ]
    response = {
        "ok": True,
        "engine": "Railway official PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec + Traditional Chinese normalization",
        "modelConfig": MODEL_CONFIG,
        "sampledFrames": len(request.frames),
        "sampledTimes": [frame.time for frame in request.frames if frame.time is not None],
        "rawText": raw_text,
        "lines": response_lines,
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
