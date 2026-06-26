import os
import cv2

# =====================================================
# PATHS
# =====================================================

INPUT_ROOT = "/kaggle/input/datasets/shizunyan/oumvlp-gei/GEI"

OUTPUT_ROOT = "data/processed/gei_oumvlp"

TARGET_SIZE = (64, 64)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# PROCESS
# =====================================================

folder_count = 0
gei_count = 0

for folder in sorted(os.listdir(INPUT_ROOT)):

    folder_path = os.path.join(INPUT_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    folder_count += 1

    for file_name in sorted(os.listdir(folder_path)):

        if not file_name.endswith(".png"):
            continue

        file_path = os.path.join(folder_path, file_name)

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # -----------------------------------------
        # PAD THÀNH VUÔNG TRƯỚC
        # -----------------------------------------

        h, w = img.shape
        pad = (h - w) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)

        # -----------------------------------------
        # RESIZE
        # -----------------------------------------

        img = cv2.resize(img, TARGET_SIZE)

        # -----------------------------------------
        # OUTPUT PATH
        # -----------------------------------------

        subject_id = os.path.splitext(file_name)[0]

        save_dir = os.path.join(OUTPUT_ROOT, subject_id)

        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{folder}.png")

        cv2.imwrite(save_path, img)

        gei_count += 1

# =====================================================
# DONE
# =====================================================

print("Done")
print("Folders processed:", folder_count)
print("GEIs saved:", gei_count)