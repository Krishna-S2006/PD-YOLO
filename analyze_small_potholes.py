"""Read-only ground-truth size analysis; no labels or images are modified."""
import argparse, csv
from pathlib import Path
import cv2, numpy as np, yaml
from evaluate_tiling import load_images

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data', default='ultralytics/cfg/datasets/Bump.yaml'); p.add_argument('--split', default='test'); p.add_argument('--small-area', type=float, default=32**2); p.add_argument('--medium-area', type=float, default=96**2); p.add_argument('--output', default='runs/analysis/object_size_distribution.csv'); a=p.parse_args()
    images,_=load_images(a.data,a.split); rows=[]
    for image_path in images:
        image=cv2.imread(str(image_path)); h,w=image.shape[:2]; label=Path(str(image_path).replace('images','labels')).with_suffix('.txt')
        if label.exists() and label.read_text().strip():
            for cls,x,y,bw,bh in np.loadtxt(label, ndmin=2):
                area=float(bw*w*bh*h); bucket='small' if area < a.small_area else ('medium' if area < a.medium_area else 'large')
                rows.append({'image':image_path.name,'class_id':int(cls),'width_px':bw*w,'height_px':bh*h,'area_px2':area,'size':bucket})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=['image','class_id','width_px','height_px','area_px2','size']); writer.writeheader(); writer.writerows(rows)
    print({name:sum(r['size']==name for r in rows) for name in ('small','medium','large')}, 'written:',out)
if __name__=='__main__': main()
