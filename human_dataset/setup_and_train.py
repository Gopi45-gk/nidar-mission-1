import os
import shutil
import yaml
from ultralytics import YOLO
import glob

def setup_and_train():
    # Source paths (current directory where script is)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Define paths
    images_pattern = os.path.join(src_dir, 'images', '*.jpg')
    image_files = glob.glob(images_pattern)
    image_files = [f.replace('\\', '/') for f in image_files]
    
    if not image_files:
        print("No images found in ./images")
        return

    # 2. Prepare Temp Directory (to avoid issues with spaces in path)
    # Using .gemini if available, else standard temp
    dest_dir = "C:\\Users\\gopik\\.gemini\\yolo_tmp"
    if os.path.exists(dest_dir):
        try:
            shutil.rmtree(dest_dir)
        except:
            pass
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    print(f"Working in temp directory: {dest_dir}")
    
    # 3. Create Split
    import random
    random.shuffle(image_files)
    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    train_txt_dst = os.path.join(dest_dir, 'train.txt').replace('\\', '/')
    val_txt_dst = os.path.join(dest_dir, 'val.txt').replace('\\', '/')
    
    with open(train_txt_dst, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_files))
        
    with open(val_txt_dst, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_files))
        
    # 4. Create data.yaml
    # Scan for classes (optional, hardcoding based on previous findings)
    # previous search found '0' and '2'.
    # We map them to generic names.
    names_list = ['class_0', 'class_1', 'class_2'] 
    
    data_config = {
        'train': train_txt_dst,
        'val': val_txt_dst,
        'names': names_list
    }
    
    yaml_dst = os.path.join(dest_dir, 'data.yaml')
    with open(yaml_dst, 'w') as f:
        yaml.dump(data_config, f)
        
    # 5. Run Training
    print("Starting YOLOv8 training...")
    original_cwd = os.getcwd()
    try:
        os.chdir(dest_dir) # Change CWD to temp dir
        model = YOLO('yolov8n.pt') 
        # Train
        model.train(data='data.yaml', epochs=10, imgsz=640)
        
        # 6. Copy Results Back
        runs_dir = os.path.join(dest_dir, 'runs')
        target_runs_dir = os.path.join(src_dir, 'runs')
        if os.path.exists(runs_dir):
            if os.path.exists(target_runs_dir):
                # Copy with timestamp or overwrite
                shutil.rmtree(target_runs_dir) # Simple overwrite for now
            shutil.copytree(runs_dir, target_runs_dir)
            print(f"Training finished. Results copied to {target_runs_dir}")
            
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        os.chdir(original_cwd)

if __name__ == '__main__':
    setup_and_train()
