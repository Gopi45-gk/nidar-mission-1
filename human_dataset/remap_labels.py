"""
Remap labels: the original dataset uses class 0 and class 2 for humans.
This script rewrites every .txt label file to use class 0 only,
so the model learns one unified "human" class.

Run ONCE after downloading:
    python remap_labels.py
"""

import os
from pathlib import Path

BASE = Path(__file__).parent.resolve()

# Classes in the original data that correspond to "human"
HUMAN_CLASSES = {0, 2}

fixed = skipped = errors = 0

for split in ("train", "val"):
    lbl_dir = BASE / "labels" / split
    if not lbl_dir.exists():
        continue
    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        try:
            lines = lbl_file.read_text().strip().splitlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls in HUMAN_CLASSES:
                    parts[0] = "0"          # remap to human
                    new_lines.append(" ".join(parts))
                # else: drop non-human annotations (none expected here)
            lbl_file.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            fixed += 1
        except Exception as e:
            print(f"Error on {lbl_file.name}: {e}")
            errors += 1

print(f"Remapped {fixed} label files  ({errors} errors, {skipped} skipped)")
print("All class 0 and class 2 annotations → class 0 (human)")

