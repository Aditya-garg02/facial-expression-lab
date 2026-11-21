"""Model, preprocessing, and checkpoint utilities for ExpressionLab."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

EMOTIONS = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
IMAGE_SIZE = 48


@dataclass(frozen=True)
class ScoreOptions:
    mirrored: bool = False
    temperature: float = 1.0


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class ExpressionNet(nn.Module):
    """Compact residual CNN for 48×48 grayscale FER images."""
    def __init__(self, num_classes: int = len(EMOTIONS)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResidualBlock(32, 32, dropout=0.05), ResidualBlock(32, 32))
        self.stage2 = nn.Sequential(ResidualBlock(32, 64, stride=2, dropout=0.10), ResidualBlock(64, 64))
        self.stage3 = nn.Sequential(ResidualBlock(64, 128, stride=2, dropout=0.15), ResidualBlock(128, 128))
        self.stage4 = nn.Sequential(ResidualBlock(128, 192, stride=2, dropout=0.20), ResidualBlock(192, 192))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(192, num_classes))

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and the final spatial features for attribution."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        features = self.stage4(x)
        return self.head(self.pool(features)), features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits


def preprocess_face(face: Image.Image | np.ndarray) -> torch.Tensor:
    """Return normalized [1, 1, 48, 48] grayscale input."""
    if isinstance(face, np.ndarray):
        if face.ndim == 2:
            image = Image.fromarray(face.astype(np.uint8), mode="L")
        else:
            image = Image.fromarray(face.astype(np.uint8)).convert("L")
    else:
        image = face.convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32) / 255.0
    values = (values - 0.5) / 0.5
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0)


def get_score_options(metadata: dict[str, Any], *, still: bool) -> ScoreOptions:
    """Use the benchmark profile for this checkpoint, when available."""
    profile = metadata.get("inference_profile", {})
    key = "temperature" if still else "fast_temperature"
    temperature = profile.get(key, 1.0)
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        return ScoreOptions()
    return ScoreOptions(mirrored=still and profile.get("selected") == "mirrored",
                        temperature=float(temperature))


@torch.inference_mode()
def expression_scores(model: ExpressionNet, batch: torch.Tensor,
                      options: ScoreOptions = ScoreOptions()) -> torch.Tensor:
    """Return seven scores per face from a batch of preprocessed crops."""
    if options.mirrored:
        forward, mirrored = model(torch.cat((batch, batch.flip((-1,))))).chunk(2)
        logits = (forward + mirrored) * 0.5
    else:
        logits = model(batch)
    return (logits / options.temperature).softmax(dim=1)


def read_inference_profile(path: Path) -> dict[str, Any] | None:
    profile_path = path.with_name("inference_profile.json")
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict) or profile.get("selected") not in ("original", "mirrored"):
            return None
        if profile.get("checkpoint_sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
            return profile
    except (OSError, ValueError):
        pass
    return None


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[ExpressionNet, dict[str, Any]]:
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ExpressionNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device, memory_format=torch.channels_last).eval()
    metadata = {key: value for key, value in checkpoint.items()
                if key not in ("model_state_dict", "inference_profile")}
    profile = read_inference_profile(path)
    if profile is not None:
        metadata["inference_profile"] = profile
    return model, metadata
