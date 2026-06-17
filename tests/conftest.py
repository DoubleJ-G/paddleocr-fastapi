from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_image_bytes() -> bytes:
    return (FIXTURES / "sample.jpeg").read_bytes()
