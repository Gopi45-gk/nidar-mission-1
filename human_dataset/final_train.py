import os
import glob
import yaml
import sys
# Set encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

from ultralytics import YOLO

def train():
    work_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    
    # Get images
    images = glob.glob(os.path.join(work_dir, 'images', '*.jpg'))
    images = [p.replace('\\', '/') for p in images]
    
    if not images:
        print("No images found!")
        return

    # Split
    split = int(len(images) * 0.8)
    train_imgs = images[:split]
    val_imgs = images[split:]
    
    # Write txt files
    train_txt = os.path.join(work_dir, 'train.txt').replace('\\', '/')
    val_txt = os.path.join(work_dir, 'val.txt').replace('\\', '/')
    
    with open(train_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_imgs))
        
    with open(val_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_imgs))
        
    # Write data.yaml
    # Note: 'names' found earlier were 0 and 2.
    data = {
        'train': train_txt,
        'val': val_txt,
        'names': {0: 'human', 1: 'unused', 2: 'class_2'}
    }
    
    yaml_path = os.path.join(work_dir, 'data.yaml').replace('\\', '/')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)
        
    print(f"Data config created at: {yaml_path}")
    
    # Train
    try:
        model = YOLO('yolov8n.pt')
        model.train(data=yaml_path, epochs=3, imgsz=640)
        print("Training successfully finished.")
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"Training failed: {e}")
        print(err_msg)
        with open(os.path.join(work_dir, 'error.log'), 'w') as f:
            f.write(err_msg)


if __name__ == '__main__':
    train()
