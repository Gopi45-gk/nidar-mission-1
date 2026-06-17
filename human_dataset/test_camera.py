import sys
from unittest.mock import MagicMock
# Hack to bypass broken SAM import in ultralytics
try:
    sys.modules['ultralytics.models.sam'] = MagicMock()
except ImportError:
    pass

from ultralytics import YOLO
import cv2
import os

def run_camera():
    # Using Custom Trained Model
    model_path = os.path.join('runs', 'detect', 'train7', 'weights', 'best.pt') 
    
    print(f"Loading custom model: {model_path}")
    model = YOLO(model_path)

    # Use the default laptop camera (index 0)
    # ip_camera_url = 'http://100.92.66.139:8080/video'
    print("Connecting to Laptop Camera...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    else:
        print("Successfully connected to Camera.")

    print("Starting inference... Press 'q' to exit.")

    # Frame skipping variables for smoother video
    frame_count = 0
    # Process every 3rd frame (skip 2) to balance load. 
    # With Nano, we could do every frame, but skipping ensures video stream stays fluid.
    skip_frames = 2 
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Resize large frames for speed (optional, good for high-res IP cams)
        if frame.shape[1] > 1080:
             frame = cv2.resize(frame, (1024, int(frame.shape[0] * 1024 / frame.shape[1])))

        # Run inference only every (skip_frames + 1) frames
        if frame_count % (skip_frames + 1) == 0:
            # High confidence (0.70) for accuracy
            results = model(frame, stream=True, conf=0.70, classes=[0], verbose=False)
            
            # Convert generator to list so we can reuse it for skipped frames
            # Note: stream=True returns a generator, so we iterate to get the result
            current_results = []
            for r in results:
                current_results.append(r)
            last_results = current_results
        
        frame_count += 1

        # Draw boxes from the most recent inference
        if last_results:
            for result in last_results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get confidence
                    conf = float(box.conf[0])
                    percentage = f"{conf * 100:.2f}%"
                    
                    # Label
                    label = f"person: {percentage}"

                    # Draw standard yellow box (BGR: 0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                    # Draw label text
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 255, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Show the frame
        cv2.imshow("Human Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_camera()
