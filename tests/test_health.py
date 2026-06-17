from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_ready_after_startup(client: TestClient):
    # The `client` fixture uses TestClient as a context manager,
    # which fires the lifespan and warms up the OCR engine.
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_root_is_404(client: TestClient):
    response = client.get("/")
    assert response.status_code == 404
