import os
import shutil
import yaml
from ultralytics import YOLO

def train_workaround():
    # Source paths
    src_dir = os.path.dirname(os.path.abspath(__file__))
    train_txt_src = os.path.join(src_dir, 'train.txt')
    val_txt_src = os.path.join(src_dir, 'val.txt')
    
    # Dest paths (temp dir without spaces ideally, or just outside the spaced folder if possible)
    # We use .gemini as we have access
    dest_dir = "C:\\Users\\gopik\\.gemini\\yolo_tmp"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    train_txt_dst = os.path.join(dest_dir, 'train.txt')
    val_txt_dst = os.path.join(dest_dir, 'val.txt')
    yaml_dst = os.path.join(dest_dir, 'data.yaml')
    
    # Copy txt files
    shutil.copy(train_txt_src, train_txt_dst)
    shutil.copy(val_txt_src, val_txt_dst)
    
    # Create valid data.yaml in dest_dir
    # Note: verify if txt files have valid absolute paths to images. (They should from previous step)
    
    data = {
        'train': train_txt_dst,
        'val': val_txt_dst,
        'names': ['class_0', 'class_1', 'class_2']
    }
    
    with open(yaml_dst, 'w') as f:
        yaml.dump(data, f)
        
    print(f"Temporary config created at: {yaml_dst}")
    with open(yaml_dst, 'r') as f:
        print(f"Config content:\n{f.read()}")
    
    # Change CWD
    os.chdir(dest_dir)
    print(f"Changed CWD to {os.getcwd()}")
    
    # Train
    model = YOLO('yolov8n.pt')
    results = model.train(data='data.yaml', epochs=3, imgsz=640)

if __name__ == '__main__':
    train_workaround()
