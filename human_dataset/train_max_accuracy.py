"""
Maximum Accuracy Training Script for NIDAR Mission 1 (VTUAV Aerial Human Detection)

Optimized for 90%+ mAP accuracy on aerial drone footage:
  - Architecture: YOLOv8-Medium (yolov8m.pt) — 25.9M parameters (over 2.3x capacity of Small)
  - Higher Resolution: imgsz=800 (56% higher pixel density to resolve tiny aerial humans)
  - Optimizer: AdamW with Cosine Annealing LR Schedule
  - Advanced Augmentation: Mosaic, MixUp, Random Perspective (simulating UAV tilt & altitude)
  - Zero Workers: rock-solid on Python 3.14
  - Saves to: runs/human_max_accuracy/weights/best.pt
"""

import sys
from pathlib import Path


def main():
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch not found.")

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅  GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    from ultralytics import YOLO

    base_dir = Path(__file__).parent.resolve()
    data_yaml = base_dir / "data.yaml"
    runs_dir = base_dir / "runs"
    model_name = "yolov8m.pt"

    print(f"\nLoading high-accuracy base model: {model_name}")
    model = YOLO(model_name)

    print("\nStarting Ultra-Accuracy Training (imgsz=800, YOLOv8-Medium, 60 epochs)…\n")
    results = model.train(
        data=str(data_yaml),
        epochs=60,
        imgsz=800,           # High-resolution for tiny overhead drone pedestrians
        device=0,            # Force RTX 3050 GPU
        amp=True,            # Mixed precision
        batch=8,             # Fits cleanly in 2.6GB VRAM
        workers=0,           # Python 3.14 safe
        patience=20,         # Generous early-stopping patience
        project=str(runs_dir),
        name="human_max_accuracy",
        exist_ok=True,

        # ── Optimizer ────────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.0008,          # Finer learning rate for Medium model
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,

        # ── Augmentations ────────────────────────────────────────────────────
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        degrees=12.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0008,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # ── Checkpointing ────────────────────────────────────────────────────
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    best_weights = runs_dir / "human_max_accuracy" / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("Ultra-Accuracy Training Complete!")
    print(f"  Best Weights : {best_weights}")
    if hasattr(results, "results_dict"):
        m50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
        m95 = results.results_dict.get("metrics/mAP50-95(B)", "N/A")
        print(f"  mAP50        : {m50}")
        print(f"  mAP50-95     : {m95}")
    print("=" * 60)


if __name__ == "__main__":
    main()

