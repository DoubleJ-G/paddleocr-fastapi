import pytest
from fastapi.testclient import TestClient


def test_missing_file_returns_422(client: TestClient):
    response = client.post("/ocr/")
    assert response.status_code == 422


def test_non_image_content_type_rejected(client: TestClient):
    response = client.post(
        "/ocr/",
        files={"file": ("hello.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
    assert "image" in response.json()["detail"].lower()


def test_corrupt_image_returns_400(client: TestClient):
    response = client.post(
        "/ocr/",
        files={"file": ("broken.jpg", b"not really a jpeg", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid image file"}


def test_oversized_upload_rejected(client: TestClient):
    payload = b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/ocr/",
        files={"file": ("big.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


@pytest.mark.slow
def test_happy_path_returns_results(client: TestClient, sample_image_bytes: bytes):
    response = client.post(
        "/ocr/",
        files={"file": ("sample.jpeg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    assert len(body["results"]) > 0
    for box in body["results"]:
        assert set(box.keys()) == {"text", "confidence", "box"}
        assert isinstance(box["text"], str)
        assert 0.0 <= box["confidence"] <= 1.0
