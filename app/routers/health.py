from fastapi import APIRouter, HTTPException

from app.ocr_engine import is_engine_ready

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    if not is_engine_ready():
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    return {"status": "ready"}
