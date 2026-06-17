# Human Detection Project

This project uses YOLOv8 to detect humans in real-time using a webcam. It supports both custom-trained models and high-accuracy pre-trained models.

## 🚀 Quick Start

To start the camera detection, run:

```bash
python test_camera.py
```

This will open a window showing your webcam feed with yellow bounding boxes around detected people.
- **Press 'q'** to stop the program.

---

## 📂 Project Structure

- **`test_camera.py`**: The main script to run the webcam detection.
- **`convert_xml_to_yolo.py`**: Tool to convert XML annotations (VOC format) to YOLO format.
- **`setup_and_train.py` / `final_train.py`**: Scripts used for training the custom model.
- **`runs/`**: Directory where training results and model weights are saved.

---

## 🧠 Models

This project currently has access to two main models:

### 1. High Accuracy Model (Recommended)
- **File:** `yolov8m.pt` (Medium version)
- **Use Case:** Best for general detection, including people sitting or in difficult poses.
- **Status:** **Active** by default in `test_camera.py`.

### 2. Custom Trained Model
- **File:** `runs/detect/train7/weights/best.pt`
- **Use Case:** Trained specifically on your dataset. Might be faster but potentially less accurate for general poses if the dataset was small.
- **Status:** Currently commented out in the code.

---

## ⚙️ Configuration

### Switching Models
To switch back to your custom trained model, edit `test_camera.py`:

```python
# Open test_camera.py and look for this section:

# model_path = 'yolov8m.pt'  <-- Comment this out
model_path = os.path.join(os.getcwd(), 'runs', 'detect', 'train7', 'weights', 'best.pt') # <-- Uncomment this
```

### Changing Camera Source
If you have multiple cameras or need to capture the screen:

```python
# In test_camera.py:
cap = cv2.VideoCapture(0)  # 0 = Default Webcam, 1 = External USB Camera
```

---

## 🔧 Troubleshooting

**Error: `videoio(MSMF): can't grab frame`**
- This means the camera is busy or disconnected.
- **Fix:** Close other apps using the camera (Zoom, Teams, etc.) and try unplugging/replugging the USB camera.

**Error: `ModuleNotFoundError: No module named 'cv2'`**
- You need to install the OpenCV library:
  ```bash
  pip install opencv-python
  ```
