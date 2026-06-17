from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logging_setup import setup_logging
from app.ocr_engine import close_ocr, get_ocr
from app.routers import health, ocr

setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Application starting up")
    get_ocr()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down")
    close_ocr()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ocr.router)
