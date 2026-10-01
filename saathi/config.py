from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 12
    recognition_threshold: float = 0.50
    recognition_margin: float = 0.08
    stable_frames: int = 5
    greeting_cooldown_seconds: float = 90
    enrollment_samples: int = 10
    enrollment_timeout_seconds: float = 35
    min_face_pixels: int = 85
    min_blur_score: float = 45
    min_brightness: float = 40
    max_brightness: float = 220
    chat_model: str = "gpt-4.1-mini"
    transcription_model: str = "gpt-4o-mini-transcribe"
    microphone_index: int | None = None
    record_seconds: int = 6
    objects_model: str | None = None

    def __post_init__(self):
        limits = {
            "camera_index": (0, 32), "frame_width": (320, 1920),
            "frame_height": (240, 1080), "target_fps": (1, 30),
            "recognition_threshold": (0.1, 1), "recognition_margin": (0, 1),
            "stable_frames": (2, 30), "greeting_cooldown_seconds": (10, 3600),
            "enrollment_samples": (5, 20), "enrollment_timeout_seconds": (10, 120),
            "min_face_pixels": (30, 400), "min_blur_score": (0, 1000),
            "min_brightness": (0, 254), "max_brightness": (1, 255),
            "record_seconds": (2, 15),
        }
        int_fields = {f.name for f in fields(self) if f.type is int}
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            if key in int_fields and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
        if self.min_brightness >= self.max_brightness:
            raise ValueError("min_brightness must be below max_brightness")
        for key in ("chat_model", "transcription_model"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip() or len(value) > 150:
                raise ValueError(f"Invalid {key}")
        if self.microphone_index is not None and (
            type(self.microphone_index) is not int or self.microphone_index < 0
        ):
            raise ValueError("microphone_index must be null or a nonnegative integer")
        if self.objects_model is not None and not isinstance(self.objects_model, str):
            raise ValueError("objects_model must be a local model path or null")

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else ROOT / "config.local.json"
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
        return cls(**values)
