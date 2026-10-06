"""Evaluate unchanged PD-YOLO weights in baseline or tiled inference mode."""
import argparse, json, time
from pathlib import Path
import cv2, numpy as np, torch, yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou
from tile_inference import Detection, TiledInference, read_image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}

def args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--data', default='ultralytics/cfg/datasets/Bump.yaml')
    p.add_argument('--split', default='test', choices=['val', 'test']); p.add_argument('--tiling', dest='tiling', action='store_true')
    p.add_argument('--no-tiling', dest='tiling', action='store_false'); p.set_defaults(tiling=False)
    p.add_argument('--tile-width', type=int, default=640); p.add_argument('--tile-height', type=int, default=640); p.add_argument('--overlap-ratio', type=float, default=.15)
    p.add_argument('--conf', type=float, default=.25); p.add_argument('--nms-iou-threshold', type=float, default=.7); p.add_argument('--device', default=None)
    p.add_argument('--output', default='runs/evaluate/metrics.json'); return p.parse_args()

def load_images(data_file, split):
    # Dataset YAML contains UTF-8 comments; Windows' locale default can be cp1252.
    data_file = Path(data_file).resolve(); data = yaml.safe_load(data_file.read_text(encoding='utf-8'))
    root = Path(data.get('path', data_file.parent)); split_path = Path(data[split]); image_dir = split_path if split_path.is_absolute() else root / split_path
    return sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES), data

def labels_for(image_path, image):
    label = Path(str(image_path).replace('images', 'labels')).with_suffix('.txt')
    if not label.exists() or not label.read_text().strip(): return torch.zeros((0, 5), dtype=torch.float32)
    rows = np.loadtxt(label, ndmin=2, dtype=np.float32); h, w = image.shape[:2]
    cls, x, y, bw, bh = rows.T
    return torch.tensor(np.column_stack((cls, (x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)), dtype=torch.float32)

def correct_predictions(pred, labels, iouv):
    correct = torch.zeros((len(pred), len(iouv)), dtype=torch.bool)
    if not len(pred) or not len(labels): return correct
    iou = box_iou(labels[:, 1:], pred[:, :4]); matches = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == pred[:, 5]))
    if matches[0].numel():
        matched = torch.cat((torch.stack(matches, 1), iou[matches[0], matches[1]][:, None]), 1).cpu().numpy()
        matched = matched[matched[:, 2].argsort()[::-1]]
        matched = matched[np.unique(matched[:, 1], return_index=True)[1]]
        matched = matched[matched[:, 2].argsort()[::-1]]
        matched = matched[np.unique(matched[:, 0], return_index=True)[1]]
        matched = torch.tensor(matched, dtype=torch.long)
        correct[matched[:, 1]] = iou[matched[:, 0], matched[:, 1]][:, None] >= iouv
    return correct

def main():
    a = args(); images, data = load_images(a.data, a.split); model = YOLO(a.weights); iouv = torch.linspace(.5, .95, 10)
    stats, durations = [], []
    tiled = TiledInference(model, a.tile_width, a.tile_height, a.overlap_ratio, a.conf, a.nms_iou_threshold, device=a.device) if a.tiling else None
    for image_path in images:
        image = read_image(image_path); started = time.perf_counter()
        if tiled:
            ds, _ = tiled.predict(image); pred = torch.tensor([[*d.xyxy, d.confidence, d.class_id] for d in ds], dtype=torch.float32).reshape(-1, 6)
        else:
            r = model.predict(image, conf=a.conf, iou=a.nms_iou_threshold, device=a.device, verbose=False)[0]
            pred = torch.cat((r.boxes.xyxy.cpu(), r.boxes.conf[:, None].cpu(), r.boxes.cls[:, None].cpu()), 1) if len(r.boxes) else torch.zeros((0, 6))
        durations.append(time.perf_counter() - started); labels = labels_for(image_path, image)
        stats.append((correct_predictions(pred, labels, iouv).cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), labels[:, 0].cpu().numpy()))
    tp, conf, pred_cls, target_cls = (np.concatenate(x, 0) for x in zip(*stats))
    _, _, precision, recall, _, ap, classes, *_ = ap_per_class(tp, conf, pred_cls, target_cls, names=data.get('names', {}))
    result = {'mode': 'tiled' if a.tiling else 'baseline', 'images': len(images), 'precision': float(precision.mean()), 'recall': float(recall.mean()),
              'mAP50': float(ap[:, 0].mean()), 'mAP50-95': float(ap.mean()), 'inference_time': float(np.mean(durations)), 'FPS': float(1 / np.mean(durations)),
              'tile_width': a.tile_width if a.tiling else None, 'tile_height': a.tile_height if a.tiling else None, 'overlap_ratio': a.overlap_ratio if a.tiling else None}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
if __name__ == '__main__': main()
