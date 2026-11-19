"""
Improved YOLOv8 training script for the NIDAR human-detection project.

Key improvements:
  - GPU-ONLY: strictly uses NVIDIA GPU (device=0)
  - Remaps classes 0 & 2 -> class 0 ("human")
  - Uses YOLOv8-Small (yolov8s.pt)
  - 50 epochs with cosine LR decay
  - Full augmentation: mosaic, mixup, copy-paste, perspective, flips, HSV jitter
  - AMP (fp16) enabled for faster GPU training
  - workers=0: completely eliminates Python 3.14 multiprocessing / forkserver bugs
"""

import os
import sys
from pathlib import Path


def main():
    # ── GPU CHECK — abort early if no CUDA GPU ───────────────────────────────
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch not found. Run: pip install torch --index-url https://download.pytorch.org/whl/cu128")

    if not torch.cuda.is_available():
        sys.exit(
            "\n❌  No CUDA GPU detected!\n"
            "   This script is configured to run on GPU only.\n"
            "   Make sure you have:\n"
            "     1. An NVIDIA GPU installed\n"
            "     2. CUDA-enabled PyTorch:  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "   Run  nvidia-smi  to verify your GPU is visible.\n"
        )

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅  GPU detected: {gpu_name}  ({vram_gb:.1f} GB VRAM)")

    # ── make sure ultralytics is importable ──────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not found. Run: pip install ultralytics")

    # ── paths ────────────────────────────────────────────────────────────────
    base_dir = Path(__file__).parent.resolve()
    data_yaml = base_dir / "data.yaml"
    model_pt = "yolov8s.pt"
    runs_dir = base_dir / "runs"

    # ── sanity-check: data must exist ────────────────────────────────────────
    for split in ("train", "val"):
        img_dir = base_dir / "images" / split
        lbl_dir = base_dir / "labels" / split
        n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        if n_img == 0:
            sys.exit(
                f"\nNo images found in {img_dir}\n"
                "Run:  python download_dataset.py   first.\n"
            )
        print(f"  {split}: {n_img} images, {n_lbl} labels")

    print(f"\nLoading base model: {model_pt}")
    model = YOLO(model_pt)

    print("\nStarting improved GPU training…\n")
    results = model.train(
        data        = str(data_yaml),
        epochs      = 50,
        imgsz       = 640,
        device      = 0,           # GPU 0 — forced
        amp         = True,        # fp16 mixed precision -> 2x faster on GPU
        batch       = 16,
        workers     = 0,           # 0 eliminates Python 3.14 forkserver issues
        patience    = 15,
        project     = str(runs_dir),
        name        = "human_v2",
        exist_ok    = True,        # save into same folder without duplicating

        # ── optimizer ────────────────────────────────────────────────────────
        optimizer   = "AdamW",
        lr0         = 0.001,
        lrf         = 0.01,
        warmup_epochs = 3,
        weight_decay  = 0.0005,

        # ── augmentation (critical for small dataset) ────────────────────────
        mosaic      = 1.0,
        mixup       = 0.1,
        copy_paste  = 0.1,
        degrees     = 10.0,
        translate   = 0.1,
        scale       = 0.5,
        shear       = 2.0,
        perspective = 0.0005,
        flipud      = 0.3,
        fliplr      = 0.5,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,

        # ── misc ─────────────────────────────────────────────────────────────
        save        = True,
        save_period = 10,
        plots       = True,
        verbose     = True,
    )

    best_weights = runs_dir / "human_v2" / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best weights : {best_weights}")
    if hasattr(results, "results_dict"):
        m50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
        m95 = results.results_dict.get("metrics/mAP50-95(B)", "N/A")
        print(f"  mAP50        : {m50}")
        print(f"  mAP50-95     : {m95}")
    print("=" * 60)
    print("\nTo test with camera, run:  python test_camera.py")


if __name__ == "__main__":
    main()
