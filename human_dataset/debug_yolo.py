import os
import yaml
from ultralytics import YOLO
import traceback

def run_debug():
    try:
        print(f"CWD: {os.getcwd()}")
        if os.path.exists('data.yaml'):
            print("data.yaml exists.")
            with open('data.yaml', 'r') as f:
                print(f"data.yaml content:\n{f.read()}")
        else:
            print("data.yaml MISSING!")
            return

        model = YOLO('yolov8n.pt')
        # Try training
        results = model.train(data='data.yaml', epochs=1, imgsz=640)
        print("Success!")
    except Exception:
        with open('error.log', 'w') as f:
            f.write(traceback.format_exc())
        print("Training Failed. Error logged to error.log")

if __name__ == '__main__':
    run_debug()
