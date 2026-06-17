# fast-api-ocr

A small FastAPI service that wraps [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) behind an HTTP endpoint.

I built this for two reasons:

1. To play with FastAPI.
2. PaddleOCR has a heavy startup cost — initializing the engine takes seconds. Invoking it as a one-shot script per image is painful. Wrapping it in a long-running service keeps the engine warm in memory and reduces per-request latency to just the inference cost.

The engine is loaded once at app startup (via FastAPI's `lifespan`) and reused across requests.

## Stack

- Python 3.12+, FastAPI, Uvicorn
- PaddleOCR 3.7 with the ONNX Runtime backend (English, detection-only — no orientation/unwarping)
- `structlog` for JSON logging
- `uv` for dependency management
- Scalar for API docs
- pytest + ruff + pyright

## Endpoints

| Method | Path     | Description                                                                   |
| ------ | -------- | ----------------------------------------------------------------------------- |
| POST   | `/ocr/`  | Multipart upload (`file=<image>`). Returns detected text, confidence, boxes.  |
| GET    | `/health`| Liveness. Always returns `{"status": "ok"}`.                                  |
| GET    | `/ready` | Readiness. 200 once the OCR engine is initialized, 503 otherwise.             |
| GET    | `/docs`  | Scalar-rendered API reference.                                                |

Response shape:

```json
{
  "results": [
    { "text": "hello", "confidence": 0.9876, "box": [[x, y], [x, y], [x, y], [x, y]] }
  ]
}
```

Large images are downscaled to `ocr_max_dimension` (default 2048px on the longest side) before inference, and box coordinates are rescaled back to the original image's space.

## Running locally

```bash
uv sync
uv run uvicorn app.main:app --reload
```

First boot will download the PaddleOCR model weights. Subsequent boots reuse the cached models.

## Running with Docker

The Dockerfile bakes the model weights into the image at build time so containers start without a network round-trip:

```bash
docker compose up --build
```

Or without compose:

```bash
docker build -t fast-api-ocr .
docker run --rm -p 8000:8000 fast-api-ocr
```

## Configuration

Environment variables (via `pydantic-settings`):

| Variable             | Default            | Purpose                                        |
| -------------------- | ------------------ | ---------------------------------------------- |
| `LOG_PRETTY`         | `false`            | `true` for human-readable console logs.        |
| `OCR_MAX_DIMENSION`  | `2048`             | Longest side (px) before downscaling.          |
| `MAX_UPLOAD_BYTES`   | `10485760` (10 MB) | Per-request upload limit.                      |

## Design notes

A few trade-offs worth calling out — they're not obvious from skimming the code.

### Concurrency

PaddleOCR inference is synchronous, CPU-bound, and takes seconds per image. Calling it directly from an `async def` handler would block the event loop, so even `/health` would hang while a single image was being processed. PaddleOCR also doesn't document its predictors as thread-safe, and there's a single shared engine cached on the app.

The compromise: inference is gated behind an `asyncio.Semaphore(1)` and offloaded via `asyncio.to_thread`. Concurrent OCR requests queue rather than race on the shared engine, and the event loop stays free to serve everything else. Horizontal scaling (multiple Uvicorn workers, one engine each) is the path to higher throughput.

Measured on `tests/fixtures/sample.jpeg` (~5s inference):

```
OCR request (background):  HTTP 200 in 5.15s
  /health probe 1: HTTP 200 in 0.0013s
  /health probe 2: HTTP 200 in 0.0015s
  ... (8 probes during the OCR window)
  /health probe 8: HTTP 200 in 0.0013s

Two concurrent OCR requests:
  B: HTTP 200 in  5.15s   ← took the semaphore first
  A: HTTP 200 in 10.39s   ← queued behind B, then ran
```

### Model choice

- **ONNX Runtime backend** rather than the default Paddle Inference runtime. Smaller install footprint, faster cold start, no GPU dependencies — the right default for a CPU-only service.
- **Detection + recognition only.** Orientation classification and document unwarping are disabled in `ocr_engine.py`, since both add latency and aren't needed for the common case (already-upright photos and screenshots).
- **Lowered detection thresholds** (`text_det_thresh=0.2`, `text_det_box_thresh=0.4`) for better recall on faint or low-contrast text. The trade is more false-positive boxes — callers can filter on `confidence` if precision matters.

### Image preprocessing

Images larger than `OCR_MAX_DIMENSION` (default 2048px) are downscaled before inference, then box coordinates are rescaled back to the original image's coordinate space. PaddleOCR's accuracy plateaus well below 4K, so feeding it full-resolution phone photos just burns CPU.

### Engine lifecycle

The engine is constructed once during FastAPI's `lifespan` startup and lives for the process's lifetime. `/ready` reports 503 until initialization completes, so orchestrators can route traffic only when the engine is actually warm.

## Tests

```bash
uv run pytest                  # fast tests only
uv run pytest -m slow          # also runs real OCR inference
```

Slow tests load the actual PaddleOCR engine and are skipped by default.
