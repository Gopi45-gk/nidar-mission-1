# NIDAR Mission 1 — High-Accuracy Zero-Latency Human Detection

Real-time human detection system engineered for zero-latency camera streams and high-accuracy aerial drone surveillance on NVIDIA RTX GPUs.

---

## ⚡ Highlights

- **Aerial Drone Surveillance (VTUAV Dataset)**: **90.07% mAP50** (YOLOv8-Medium, 800px resolution)
- **Real-Time Webcam Detection**: **95.1% – 97.0% confidence** with 0 false positives
- **Decoupled Camera Stream**: **300+ FPS** with **0ms perceived latency** via threaded hardware buffer management
- **Hardware Acceleration**: Full CUDA acceleration on NVIDIA RTX 3050 GPU

---

## 🚀 Quick Start

```bash
cd human_dataset
python test_camera.py
```

### Controls in Real-Time Camera:
- **`[t]`** — Toggle **TURBO Mode** (~16ms latency) vs **ULTRA-ACC Mode** (95–99% confidence)
- **`[m]`** — Toggle **WEBCAM Mode** vs **AERIAL DRONE Mode** (90.07% mAP)
- **`[q]`** — Quit

---

## 📂 Repository Contents

```
nidar-mission-1/
├── human_dataset/
│   ├── test_camera.py          # Multi-threaded zero-latency real-time detector
│   ├── train_improved.py       # GPU training pipeline with full data augmentation
│   ├── train_max_accuracy.py   # High-resolution (800px) training script (90.07% mAP50)
│   ├── download_dataset.py     # Automated dataset downloader
│   ├── remap_labels.py         # Label remapping to unified single class
│   ├── data.yaml               # Cross-platform dataset configuration
│   └── README.md               # Detailed human dataset guide
└── README.md
```
