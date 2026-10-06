"""Run baseline and 2x2-style overlapping tiling experiments and collect JSON/CSV."""
import argparse, csv, json, subprocess, sys
from pathlib import Path

EXPERIMENTS=[('baseline',False,0.0),('tile_2x2_no_overlap',True,0.0),('tile_2x2_10pct',True,.10),('tile_2x2_15pct',True,.15),('tile_2x2_20pct',True,.20)]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',required=True); p.add_argument('--data',default='ultralytics/cfg/datasets/Bump.yaml'); p.add_argument('--split',default='test'); p.add_argument('--tile-width',type=int,default=640); p.add_argument('--tile-height',type=int,default=640); p.add_argument('--conf',type=float,default=.25); p.add_argument('--nms-iou-threshold',type=float,default=.7); p.add_argument('--output-dir',default='runs/ablation'); a=p.parse_args()
    root=Path(__file__).parent; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); results=[]
    for name,tiled,overlap in EXPERIMENTS:
        metrics=out/f'{name}.json'; command=[sys.executable,str(root/'evaluate_tiling.py'),'--weights',a.weights,'--data',a.data,'--split',a.split,'--conf',str(a.conf),'--nms-iou-threshold',str(a.nms_iou_threshold),'--output',str(metrics)]
        if tiled: command += ['--tiling','--tile-width',str(a.tile_width),'--tile-height',str(a.tile_height),'--overlap-ratio',str(overlap)]
        subprocess.run(command,check=True); record=json.loads(metrics.read_text()); record['configuration']=name; results.append(record)
    with (out/'ablation_results.csv').open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=sorted({k for r in results for k in r})); writer.writeheader(); writer.writerows(results)
    (out/'ablation_results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__': main()
