
import os
import numpy as np
import cv2

# =====================================================
# PATHS
# =====================================================

# Trỏ vào folder GEI đã preprocess sẵn (64x64)
INPUT_ROOT = r"data/processed/gei_oumvlp"

OUTPUT_DIR = r"data/npy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# COLLECT
# =====================================================

images = []
labels = []

for subject in sorted(os.listdir(INPUT_ROOT)):

    subject_path = os.path.join(INPUT_ROOT, subject)

    if not os.path.isdir(subject_path):
        continue

    label = int(subject) - 1

    for file_name in sorted(os.listdir(subject_path)):

        if not file_name.endswith(".png"):
            continue

        img = cv2.imread(
            os.path.join(subject_path, file_name),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        images.append(img)
        labels.append(label)

# =====================================================
# SAVE
# =====================================================

images_arr = np.array(images, dtype=np.uint8)   # uint8 để nhẹ file
labels_arr = np.array(labels, dtype=np.int32)

np.save(os.path.join(OUTPUT_DIR, "oumvlp_images.npy"), images_arr)
np.save(os.path.join(OUTPUT_DIR, "oumvlp_labels.npy"), labels_arr)

print("Done")
print(f"Images: {images_arr.shape}")
print(f"Labels: {labels_arr.shape}")