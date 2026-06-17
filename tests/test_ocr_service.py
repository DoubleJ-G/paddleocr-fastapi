import io

import pytest
from fastapi import HTTPException
from PIL import Image

from app.config import settings
from app.ocr_service import _format_results, _load_image, _resize_if_needed


def test_resize_skipped_when_image_within_limit():
    img = Image.new("RGB", (800, 600))

    result, scale = _resize_if_needed(img)

    assert result is img
    assert scale == 1.0


def test_resize_applied_when_image_exceeds_limit():
    oversized = settings.ocr_max_dimension * 2
    img = Image.new("RGB", (oversized, oversized // 2))

    result, scale = _resize_if_needed(img)

    assert result is not img
    assert max(result.size) == settings.ocr_max_dimension
    assert scale == 0.5


def test_resize_preserves_aspect_ratio():
    img = Image.new("RGB", (settings.ocr_max_dimension * 3, settings.ocr_max_dimension))
    original_ratio = img.size[0] / img.size[1]

    result, _ = _resize_if_needed(img)
    new_ratio = result.size[0] / result.size[1]

    assert abs(original_ratio - new_ratio) < 0.01


def test_load_image_accepts_valid_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(buf, format="PNG")

    img = _load_image(buf.getvalue())

    assert img.size == (32, 32)
    assert img.mode == "RGB"


def test_load_image_rejects_garbage():
    with pytest.raises(HTTPException) as exc:
        _load_image(b"definitely not an image")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid image file"


def test_format_results_scales_boxes_back_to_original():
    fake_result = [
        {
            "rec_texts": ["hello"],
            "rec_scores": [0.95],
            "rec_polys": [[[100, 100], [200, 100], [200, 150], [100, 150]]],
        }
    ]

    boxes = _format_results(fake_result, scale=0.5)

    assert len(boxes) == 1
    box = boxes[0]
    assert box.text == "hello"
    assert box.confidence == 0.95
    assert box.box == [[200, 200], [400, 200], [400, 300], [200, 300]]


def test_format_results_handles_empty_page():
    fake_result = [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]

    boxes = _format_results(fake_result, scale=1.0)

    assert boxes == []


def test_format_results_rounds_confidence_to_four_places():
    fake_result = [
        {
            "rec_texts": ["x"],
            "rec_scores": [0.987654321],
            "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
        }
    ]

    boxes = _format_results(fake_result, scale=1.0)

    assert boxes[0].confidence == 0.9877
