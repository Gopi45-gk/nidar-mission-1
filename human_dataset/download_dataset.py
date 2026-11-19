"""
Download all images and labels from GitHub and organise them into
the proper YOLO folder structure:

  human_dataset/
    images/
      train/   (80% of data)
      val/     (20% of data)
    labels/
      train/
      val/

Run once before training:
    python download_dataset.py
"""

import os, random, urllib.request, urllib.error

REPO   = "https://raw.githubusercontent.com/Gopi45-gk/nidar-mission-1/master/human_dataset"
TOTAL  = 512
SEED   = 42
BASE   = os.path.dirname(os.path.abspath(__file__))

for d in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)

indices = [f"{i:05d}" for i in range(1, TOTAL + 1)]
random.seed(SEED)
random.shuffle(indices)
split   = int(TOTAL * 0.8)
train_i = set(indices[:split])
val_i   = set(indices[split:])

def download(url, dst):
    if os.path.exists(dst):
        return  # already downloaded
    try:
        urllib.request.urlretrieve(url, dst)
    except urllib.error.HTTPError as e:
        print(f"  SKIP {url}: {e}")

print(f"Downloading {TOTAL} images and labels...")
for idx in sorted(indices):
    split_name = "train" if idx in train_i else "val"

    img_url = f"{REPO}/images/{idx}.jpg"
    img_dst = os.path.join(BASE, "images", split_name, f"{idx}.jpg")
    download(img_url, img_dst)

    lbl_url = f"{REPO}/labels/{idx}.txt"
    lbl_dst = os.path.join(BASE, "labels", split_name, f"{idx}.txt")
    download(lbl_url, lbl_dst)

# Count results
for split_name in ("train", "val"):
    n_img = len([f for f in os.listdir(os.path.join(BASE, "images", split_name)) if f.endswith(".jpg")])
    n_lbl = len([f for f in os.listdir(os.path.join(BASE, "labels", split_name)) if f.endswith(".txt")])
    print(f"  {split_name}: {n_img} images, {n_lbl} labels")

print("\nDone! Now run:  python train_improved.py")
