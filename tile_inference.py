"""Optional overlapping-image tiling for unchanged PD-YOLO weights.

This module never writes tiles or labels to the dataset.  A tile is an in-memory
view of the original image; its prediction is translated back before one global,
class-aware NMS removes duplicates introduced by overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class Tile:
    image: np.ndarray
    x: int
    y: int


@dataclass
class Detection:
    xyxy: np.ndarray  # float32 [x1, y1, x2, y2], original-image coordinates
    confidence: float
    class_id: int


class TileGenerator:
    """Generate full-coverage, overlapping in-memory tiles without resizing."""

    def __init__(self, tile_width: int = 640, tile_height: int = 640, overlap_ratio: float = 0.15):
        if tile_width < 1 or tile_height < 1:
            raise ValueError("tile dimensions must be positive")
        if not 0 <= overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in [0, 1)")
        self.tile_width, self.tile_height, self.overlap_ratio = tile_width, tile_height, overlap_ratio

    @staticmethod
    def _starts(length: int, tile_length: int, stride: int) -> list[int]:
        """Return starts that cover the full dimension using the configured stride."""
        if length <= tile_length:
            return [0]
        return list(range(0, length, stride))

    def generate(self, image: np.ndarray) -> list[Tile]:
        if image is None or image.ndim < 2:
            raise ValueError("image must be a non-empty HxW array")
        height, width = image.shape[:2]
        stride_x = max(1, round(self.tile_width * (1 - self.overlap_ratio)))
        stride_y = max(1, round(self.tile_height * (1 - self.overlap_ratio)))
        return [
            Tile(image[y:min(y + self.tile_height, height), x:min(x + self.tile_width, width)], x, y)
            for y in self._starts(height, self.tile_height, stride_y)
            for x in self._starts(width, self.tile_width, stride_x)
        ]


def clip_and_translate(boxes: np.ndarray, tile_x: int, tile_y: int, image_width: int, image_height: int) -> np.ndarray:
    """Convert tile-local xyxy boxes to clipped original-image xyxy coordinates."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] += tile_x
    boxes[:, [1, 3]] += tile_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_height)
    return boxes


def _iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(one[0], many[:, 0]); inter_y1 = np.maximum(one[1], many[:, 1])
    inter_x2 = np.minimum(one[2], many[:, 2]); inter_y2 = np.minimum(one[3], many[:, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    union = ((one[2] - one[0]) * (one[3] - one[1]) +
             (many[:, 2] - many[:, 0]) * (many[:, 3] - many[:, 1]) - inter)
    return inter / np.maximum(union, 1e-9)


def global_nms(detections: Iterable[Detection], iou_threshold: float = 0.7) -> list[Detection]:
    """Standard class-aware NMS after all tile coordinates have been restored."""
    detections = list(detections)
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    kept: list[Detection] = []
    for class_id in sorted({d.class_id for d in detections}):
        group = [d for d in detections if d.class_id == class_id]
        boxes = np.asarray([d.xyxy for d in group], dtype=np.float32)
        order = np.argsort([-d.confidence for d in group])
        while len(order):
            current = order[0]
            kept.append(group[current])
            order = order[1:]
            if len(order):
                order = order[_iou(boxes[current], boxes[order]) <= iou_threshold]
    return sorted(kept, key=lambda d: d.confidence, reverse=True)


class TiledInference:
    """Reuse one loaded Ultralytics PD-YOLO model over original-resolution tiles."""

    def __init__(self, model, tile_width: int = 640, tile_height: int = 640, overlap_ratio: float = 0.15,
                 confidence_threshold: float = 0.25, nms_iou_threshold: float = 0.7, **predict_kwargs):
        self.model = model
        self.generator = TileGenerator(tile_width, tile_height, overlap_ratio)
        self.confidence_threshold, self.nms_iou_threshold = confidence_threshold, nms_iou_threshold
        self.predict_kwargs = predict_kwargs

    def predict(self, image: np.ndarray) -> tuple[list[Detection], list[Tile]]:
        height, width = image.shape[:2]
        tiles = self.generator.generate(image)
        detections: list[Detection] = []
        for tile in tiles:  # deliberately sequential first: model is loaded only once
            result = self.model.predict(tile.image, conf=self.confidence_threshold,
                                        iou=self.nms_iou_threshold, verbose=False, **self.predict_kwargs)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = clip_and_translate(result.boxes.xyxy.cpu().numpy(), tile.x, tile.y, width, height)
            for box, confidence, class_id in zip(boxes, result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if box[2] > box[0] and box[3] > box[1]:
                    detections.append(Detection(box, float(confidence), int(class_id)))
        return global_nms(detections, self.nms_iou_threshold), tiles


def draw_tiles_and_detections(image: np.ndarray, tiles: Iterable[Tile], detections: Iterable[Detection], names: Union[dict, list],
                              draw_tiles: bool = False) -> np.ndarray:
    canvas = image.copy()
    if draw_tiles:
        for tile in tiles:
            h, w = tile.image.shape[:2]
            cv2.rectangle(canvas, (tile.x, tile.y), (tile.x + w - 1, tile.y + h - 1), (255, 180, 0), 2)
    for d in detections:
        x1, y1, x2, y2 = d.xyxy.astype(int)
        name = names[d.class_id] if isinstance(names, (list, tuple)) else names.get(d.class_id, str(d.class_id))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"{name} {d.confidence:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
    return canvas


def read_image(path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image
