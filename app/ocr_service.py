import io
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import structlog
from fastapi import HTTPException
from PIL import Image

from app.config import settings
from app.ocr_engine import get_ocr
from app.schemas import OCRBox

logger = structlog.get_logger(__name__)


@contextmanager
def _timer() -> Iterator[list[float]]:
    start = time.monotonic()
    elapsed: list[float] = []
    yield elapsed
    elapsed.append(time.monotonic() - start)


def _load_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, Image.UnidentifiedImageError) as e:
        raise HTTPException(status_code=400, detail="Invalid image file") from e


def _resize_if_needed(img: Image.Image) -> tuple[Image.Image, float]:
    original_size = img.size
    if max(original_size) <= settings.ocr_max_dimension:
        return img, 1.0

    scale = settings.ocr_max_dimension / max(original_size)
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    logger.info(
        "Image resized",
        original_size=original_size,
        new_size=new_size,
    )
    return img, scale


def _run_prediction(img: Image.Image) -> list[dict[str, Any]]:
    try:
        ocr = get_ocr()
    except Exception as e:
        logger.exception("OCR engine initialization failed")
        raise HTTPException(status_code=500, detail="OCR engine initialization failed") from e

    try:
        return ocr.predict(np.array(img))
    except Exception as e:
        logger.exception("OCR prediction failed")
        raise HTTPException(status_code=500, detail="OCR processing failed") from e


def _format_results(result: list[dict[str, Any]], scale: float) -> list[OCRBox]:
    outputs = []
    for page in result:
        for text, score, poly in zip(
            page["rec_texts"], page["rec_scores"], page["rec_polys"], strict=True
        ):
            outputs.append(
                OCRBox(
                    text=text,
                    confidence=round(float(score), 4),
                    box=[[int(x / scale), int(y / scale)] for x, y in poly],
                )
            )
    return outputs


def process_ocr(image_bytes: bytes) -> list[OCRBox]:
    img = _load_image(image_bytes)
    img, scale = _resize_if_needed(img)

    with _timer() as elapsed:
        result = _run_prediction(img)

    outputs = _format_results(result, scale)

    logger.info(
        "OCR completed",
        texts_found=len(outputs),
        ocr_elapsed_ms=round(elapsed[0] * 1000, 2),
        texts=[o.text for o in outputs],
    )
    return outputs
