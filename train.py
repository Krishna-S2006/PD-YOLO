import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/YOLOv8n+C3_DCN+DCSP+attention.yaml')
    # model.load('yolov8n.pt')lytics/cfg/ # loading pretrain weights
    model.train(
                data='ultralytics/cfg/datasets/Bump.yaml',
                cache=False,
                imgsz=640,
                epochs=50,#lowered epoch to test the model
                batch=16,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                amp=False,  # ← ADD THIS LINE
                # patience=0, # close earlystop
                # resume='', # last.pt path
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )