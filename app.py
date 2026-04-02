from __future__ import annotations

import io
from typing import Any

from flask import Flask, jsonify, request
from PIL import Image

from config import load_config
from segmentation_utils import (
    SegmentationPredictor,
    compute_dice_score,
    compute_iou_score,
    decode_image_payload,
    encode_mask_png,
    load_mask_image,
)


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    config = load_config()
    app = Flask(__name__)
    app.config.update(config.as_flask_dict())

    if test_config:
        app.config.update(test_config)

    predictor = app.config.get("PREDICTOR")
    if predictor is None:
        predictor = SegmentationPredictor(
            checkpoint_path=app.config["MODEL_CHECKPOINT"],
            image_size=int(app.config["IMAGE_SIZE"]),
            device=app.config["DEVICE"],
            use_pretrained_backbone=bool(app.config["USE_PRETRAINED_BACKBONE"]),
        )
        app.config["PREDICTOR"] = predictor

    @app.before_request
    def enforce_token() -> Any:
        api_token = app.config.get("API_TOKEN")
        if not api_token:
            return None

        if request.endpoint in {"health", None}:
            return None

        provided_token = request.headers.get("X-API-Token")
        if provided_token != api_token:
            return jsonify({"error": "Unauthorized"}), 401

        return None

    @app.get("/health")
    def health() -> tuple[dict[str, Any], int]:
        return (
            {
                "status": "ok",
                "model": "deeplabv3_resnet50",
                "checkpoint": app.config["MODEL_CHECKPOINT"],
                "image_size": app.config["IMAGE_SIZE"],
            },
            200,
        )

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        image = decode_image_payload(request)
        if image is None:
            return (
                {
                    "error": (
                        "Send an image as multipart/form-data with the 'image' field or as JSON with "
                        "'image_base64' or 'image_path'."
                    )
                },
                400,
            )

        predictor = app.config["PREDICTOR"]
        predicted_mask = predictor.predict(image)

        response: dict[str, Any] = {
            "model": "deeplabv3_resnet50",
            "checkpoint": app.config["MODEL_CHECKPOINT"],
            "image_size": app.config["IMAGE_SIZE"],
            "mask_shape": list(predicted_mask.shape),
            "foreground_ratio": float(predicted_mask.mean()),
            "predicted_mask_png": encode_mask_png(predicted_mask),
        }

        ground_truth_image = request.files.get("ground_truth")
        if ground_truth_image is not None:
            ground_truth_mask = load_mask_image(Image.open(io.BytesIO(ground_truth_image.read())))
            response["iou"] = compute_iou_score(predicted_mask, ground_truth_mask)
            response["dice"] = compute_dice_score(predicted_mask, ground_truth_mask)

        return jsonify(response), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
