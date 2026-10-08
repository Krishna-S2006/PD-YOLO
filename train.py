import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    model = YOLO(project_dir / 'ultralytics/cfg/models/YOLOv8n+C3_DCN+DCSP+attention.yaml')
    # model.load('yolov8n.pt')lytics/cfg/ # loading pretrain weights
    model.train(
                data=str(project_dir / 'ultralytics/cfg/datasets/Bump.yaml'),
                cache=False,
                imgsz=640,
                epochs=30,#lowered epoch to test the model
                batch=4,
                close_mosaic=0,
                # Avoid Windows dataloader subprocess crashes (native segfault).
                workers=0,
                device='0',
                pretrained=False,
                optimizer='SGD', # using SGD
                amp=False,  # ← ADD THIS LINE
                # patience=0, # close earlystop
                # resume='', # last.pt path
                # fraction=0.2,
                project=str(project_dir / 'runs/train'),
                name='exp',
                )
