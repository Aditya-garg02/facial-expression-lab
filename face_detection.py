"""Local face detection and short-lived geometric tracks for the workbench."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

Box = tuple[int, int, int, int]
YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    """A YuNet instance belongs to a single image request or webcam stream."""

    def __init__(self, max_side: int):
        self.max_side = max_side
        self.input_size = (320, 320)
        self.detector = cv2.FaceDetectorYN_create(str(YUNET_MODEL), "", self.input_size, .8, .3, 500)

    def detect(self, bgr: np.ndarray) -> list[Box]:
        height, width = bgr.shape[:2]
        if min(height, width) < 24:
            return []

        scale = min(1., self.max_side / max(height, width))
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        source = bgr if scale == 1 else cv2.resize(bgr, resized, interpolation=cv2.INTER_AREA)
        if resized != self.input_size:
            self.detector.setInputSize(resized)
            self.input_size = resized
        _, faces = self.detector.detect(source)
        if faces is None:
            return []

        x_scale, y_scale = width / resized[0], height / resized[1]
        boxes = []
        for x, y, w, h in faces[:, :4]:
            left = max(0, round(float(x) * x_scale))
            top = max(0, round(float(y) * y_scale))
            right = min(width, round(float(x + w) * x_scale))
            bottom = min(height, round(float(y + h) * y_scale))
            if min(right - left, bottom - top) >= 24:
                boxes.append((left, top, right - left, bottom - top))
        return sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)


def crop_face(rgb: np.ndarray, box: Box, padding: float) -> np.ndarray:
    x, y, width, height = box
    pad_x = round(width * padding)
    pad_y = round(height * padding)
    return rgb[max(0, y - pad_y):y + height + pad_y, max(0, x - pad_x):x + width + pad_x]


def intersection_over_union(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = width * height
    return intersection / (aw * ah + bw * bh - intersection)


@dataclass
class _Track:
    box: Box
    scores: np.ndarray
    updated_at: float
    missed_frames: int = 0


class FaceScoreSmoother:
    """Associate nearby boxes, then smooth each track independently."""

    def __init__(self):
        self.tracks: list[_Track] = []

    def update(self, boxes: list[Box], scores: np.ndarray) -> np.ndarray:
        now = monotonic()
        previous = [track for track in self.tracks if now - track.updated_at < .3 and track.missed_frames < 3]
        overlaps = [
            (intersection_over_union(box, track.box), i, j)
            for i, box in enumerate(boxes)
            for j, track in enumerate(previous)
        ]

        current = [_Track(box, row, now) for box, row in zip(boxes, scores)]
        matched_current, matched_previous = set(), set()
        for overlap, i, j in sorted(overlaps, reverse=True):
            if overlap < .28:
                break
            if i in matched_current or j in matched_previous:
                continue
            track = previous[j]
            weight = exp(-max(.016, now - track.updated_at) / .14)
            current[i].scores = weight * track.scores + (1 - weight) * current[i].scores
            matched_current.add(i)
            matched_previous.add(j)

        self.tracks = current.copy()
        for j, track in enumerate(previous):
            if j not in matched_previous:
                track.missed_frames += 1
                self.tracks.append(track)
        return np.stack([track.scores for track in current]) if current else scores
