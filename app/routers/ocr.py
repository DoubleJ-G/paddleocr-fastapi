import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.ocr_service import process_ocr
from app.schemas import OCRResponse

logger = structlog.get_logger(__name__)

router = APIRouter()

_CHUNK_SIZE = 64 * 1024


@router.post("/ocr/", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    logger.info(
        "OCR request received",
        filename=file.filename,
        content_type=file.content_type,
    )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type; must be an image")

    contents = bytearray()
    try:
        while chunk := await file.read(_CHUNK_SIZE):
            contents.extend(chunk)
            if len(contents) > settings.max_upload_bytes:
                limit_mb = settings.max_upload_bytes // 1024 // 1024
                raise HTTPException(status_code=413, detail=f"File too large (max {limit_mb} MB)")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to read uploaded file", filename=file.filename)
        raise HTTPException(status_code=400, detail="Could not read file") from e

    outputs = process_ocr(bytes(contents))
    return OCRResponse(results=outputs)
