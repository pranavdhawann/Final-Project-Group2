import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def normalize_slice(img):
    """Your existing percentile-normalizer that returns a uint8 array 0–255."""
    p1, p99 = np.percentile(img, [1, 99])
    clipped = np.clip(img, p1, p99)
    normed = (clipped - p1) / max((p99 - p1), 1e-6)
    return (normed * 255).astype(np.uint8)

def preprocess_cryoet_slices(src_root, dst_dir, target_size=(128, 512, 512)):
    """Process cryoET slices and save to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    tomo_folders = [f for f in os.listdir(src_root)
                    if os.path.isdir(os.path.join(src_root, f))]
    for tomo_id in tqdm(tomo_folders, desc="Preprocessing cryoET"):
        tomo_path = os.path.join(src_root, tomo_id)
        slices = sorted([f for f in os.listdir(tomo_path)
                         if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))])
        for sf in slices:
            slice_path = os.path.join(tomo_path, sf)
            try:
                img = Image.open(slice_path)
                arr = np.array(img)
                normed = normalize_slice(arr)
                resized = Image.fromarray(normed).resize(
                    (target_size[2], target_size[1]), Image.LANCZOS
                )
                base = os.path.splitext(sf)[0]
                parts = base.split('_')
                idx = int(parts[-1]) if parts[-1].isdigit() else 0
                out_name = f"{tomo_id}_slice_{idx:04d}.jpg"
                resized.save(os.path.join(dst_dir, out_name))
            except Exception as e:
                print(f"Error processing {slice_path}: {e}")
    print("cryoET preprocessing complete.")

#Usage 
NEW_SLICES_ROOT = "/path/to/raw_tomogram_slices"
new_processed_dir = "/path/to/new_processed_dir"

preprocess_cryoet_slices(
    src_root=NEW_SLICES_ROOT,
    dst_dir=new_processed_dir,
    target_size=(128, 512, 512)
)
