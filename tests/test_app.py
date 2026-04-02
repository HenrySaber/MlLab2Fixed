from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app


class DummyPredictor:
    def predict(self, image: Image.Image) -> np.ndarray:
        return np.zeros((image.height, image.width), dtype=np.uint8)


def test_health_endpoint() -> None:
    app = create_app({"TESTING": True, "PREDICTOR": DummyPredictor(), "API_TOKEN": ""})
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_endpoint_accepts_image_upload() -> None:
    app = create_app({"TESTING": True, "PREDICTOR": DummyPredictor(), "API_TOKEN": ""})
    client = app.test_client()

    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        data={"image": (buffer, "sample.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mask_shape"] == [16, 16]
    assert payload["foreground_ratio"] == 0.0