from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    secret_key: str
    api_token: str
    model_checkpoint: str
    image_size: int
    device: str
    use_pretrained_backbone: bool

    def as_flask_dict(self) -> dict[str, object]:
        return {
            "SECRET_KEY": self.secret_key,
            "API_TOKEN": self.api_token,
            "MODEL_CHECKPOINT": self.model_checkpoint,
            "IMAGE_SIZE": self.image_size,
            "DEVICE": self.device,
            "USE_PRETRAINED_BACKBONE": self.use_pretrained_backbone,
        }


def load_config() -> AppConfig:
    load_dotenv()

    return AppConfig(
        secret_key=os.getenv("SECRET_KEY", "dev-secret-key"),
        api_token=os.getenv("API_TOKEN", ""),
        model_checkpoint=os.getenv("MODEL_CHECKPOINT", "checkpoints/house_segmentation.pth"),
        image_size=int(os.getenv("IMAGE_SIZE", "256")),
        device=os.getenv("DEVICE", "cpu"),
        use_pretrained_backbone=_as_bool(os.getenv("USE_PRETRAINED_BACKBONE"), default=False),
    )