from functools import lru_cache

import structlog
from paddleocr import PaddleOCR

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_ocr() -> PaddleOCR:
    logger.info("Initializing OCR engine")
    return PaddleOCR(
        lang="en",
        engine="onnxruntime",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_thresh=0.2,
        text_det_box_thresh=0.4,
    )


def is_engine_ready() -> bool:
    return get_ocr.cache_info().currsize > 0


def close_ocr() -> None:
    if is_engine_ready():
        get_ocr().close()
        get_ocr.cache_clear()
