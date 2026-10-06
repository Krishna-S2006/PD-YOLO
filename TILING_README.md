# Overlapping high-resolution tiling (PD-YOLO unchanged)

This experiment implements the research contribution: **overlapping high-resolution image tiling for improving small and distant pothole detection in PD-YOLO**. It is test/validation/inference preprocessing only: no dataset image, annotation, training routine, or PD-YOLO module (C3D, DCSP, BRA, backbone, neck, or head) is modified.

## Inference

From `PD-YOLO`, baseline inference remains available:

```powershell
python predict.py --weights runs/train/exp8/weights/best.pt --source dataset/VOC2007/test/images --no-tiling
```

Tiled inference (640x640, 15% overlap) uses one model instance, restores tile-local boxes to original-image coordinates, then applies global class-aware standard NMS:

```powershell
python predict.py --weights runs/train/exp8/weights/best.pt --source dataset/VOC2007/test/images --tiling --tile-width 640 --tile-height 640 --overlap-ratio 0.15 --conf 0.25 --nms-iou-threshold 0.70 --save-tile-boundaries
```

The output directory contains final detections and, when requested, `_tiles` visualizations. Tiles are in-memory crops at their native dimensions, including an anchored final tile at each right/bottom boundary; they are never saved into or substituted for the original dataset.

Create one presentation-ready three-panel demo (original / tile grid / merged detections):

```powershell
python tiling_demo.py --weights runs/train/exp8/weights/best.pt --source dataset/VOC2007/test/images/image_0007.jpg --output runs/demo/tiling_demo.jpg
```

## Evaluation and ablation

```powershell
python evaluate_tiling.py --weights runs/train/exp8/weights/best.pt --data ultralytics/cfg/datasets/Bump.yaml --split test --no-tiling --output runs/evaluate/baseline.json
python evaluate_tiling.py --weights runs/train/exp8/weights/best.pt --data ultralytics/cfg/datasets/Bump.yaml --split test --tiling --overlap-ratio 0.15 --output runs/evaluate/tiled_15.json
python run_tiling_ablation.py --weights runs/train/exp8/weights/best.pt --data ultralytics/cfg/datasets/Bump.yaml --split test
python analyze_small_potholes.py --data ultralytics/cfg/datasets/Bump.yaml --split test
python -m pytest tests/test_tile_inference.py
```

The evaluator writes precision, recall, mAP50, mAP50-95, mean end-to-end per-image inference time, and FPS to JSON. The ablation runner produces one JSON per experiment plus `ablation_results.csv` and `.json`. It evaluates baseline, no-overlap, 10%, 15%, and 20% overlap. The size analysis is read-only and writes object-level original-pixel box areas classified with configurable COCO-style thresholds: small `<32^2`, medium `<96^2`, large otherwise.

The scripts do not claim a gain: run the paired test split experiments to answer whether recall/localization of small potholes improves enough to justify the FPS cost. Sequential inference is intentionally used first for clarity and reproducibility. A later optimization may batch same-shaped tiles, but this does not alter the experimental pipeline.

Training-time tiling is intentionally out of scope for this first experiment. If pursued later, add an on-the-fly train transform that crops the image and transforms only its temporary in-memory targets; keep a no-tiling baseline and do not rewrite source labels.
