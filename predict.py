"""PD-YOLO inference launcher; --tiling preserves a switchable baseline."""
import argparse
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

from tile_inference import TiledInference, draw_tiles_and_detections, read_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--source', required=True)
    p.add_argument('--output-dir', default='runs/predict/tiling')
    p.add_argument('--tiling', dest='tiling', action='store_true', help='enable overlapping in-memory tiling')
    p.add_argument('--no-tiling', dest='tiling', action='store_false', help='use unchanged full-image baseline')
    p.set_defaults(tiling=False)
    p.add_argument('--tile-width', type=int, default=640); p.add_argument('--tile-height', type=int, default=640)
    p.add_argument('--overlap-ratio', type=float, default=.15); p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--nms-iou-threshold', type=float, default=.7); p.add_argument('--device', default=None)
    p.add_argument('--save-tile-boundaries', action='store_true')
    return p.parse_args()


def image_paths(source: str):
    source_path = Path(source)
    if source_path.is_dir():
        return [p for p in source_path.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    return [source_path]


def main():
    args = parse_args(); model = YOLO(args.weights); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for path in image_paths(args.source):
        image = read_image(path); started = time.perf_counter()
        if args.tiling:
            runner = TiledInference(model, args.tile_width, args.tile_height, args.overlap_ratio, args.conf,
                                    args.nms_iou_threshold, device=args.device)
            detections, tiles = runner.predict(image)
            rendered = draw_tiles_and_detections(image, tiles, detections, model.names)
            if args.save_tile_boundaries:
                cv2.imwrite(str(output / f'{path.stem}_tiles{path.suffix}'), draw_tiles_and_detections(image, tiles, [], model.names, True))
        else:
            result = model.predict(image, conf=args.conf, iou=args.nms_iou_threshold, device=args.device, verbose=False)[0]
            rendered = result.plot(); tiles = []
        elapsed = time.perf_counter() - started
        cv2.imwrite(str(output / path.name), rendered)
        print(f'{path.name}: mode={"tiled" if args.tiling else "baseline"}, tiles={len(tiles)}, time={elapsed:.3f}s, FPS={1/elapsed:.2f}')


if __name__ == '__main__': main()
