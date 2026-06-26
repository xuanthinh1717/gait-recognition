import os
import cv2

# =====================================================
# PATHS
# =====================================================

# Kaggle: dataset được mount tại /kaggle/input/<tên-dataset>/
# Đổi "oumvlp-gei" thành tên dataset bạn đặt khi upload lên Kaggle
INPUT_ROOT = "/kaggle/input/oumvlp-gei/GEI"

OUTPUT_ROOT = "data/processed/gei_oumvlp"

TARGET_SIZE = (64, 64)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# PROCESS
# =====================================================

folder_count = 0
gei_count = 0

for folder in sorted(os.listdir(INPUT_ROOT)):

    # folder = "000-00", "000-01", "015-00", ...
    folder_path = os.path.join(INPUT_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nFolder: {folder}")

    folder_count += 1

    for file_name in sorted(os.listdir(folder_path)):

        if not file_name.endswith(".png"):
            continue

        # -----------------------------------------
        # READ
        # -----------------------------------------

        file_path = os.path.join(folder_path, file_name)

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # -----------------------------------------
        # RESIZE
        # -----------------------------------------

        img = cv2.resize(img, TARGET_SIZE)

        # -----------------------------------------
        # OUTPUT PATH
        # OU-MVLP không có folder per subject
        # → giữ nguyên cấu trúc: subject_id/angle-trial.png
        # để create_split xử lý giống CASIA-B
        # -----------------------------------------

        subject_id = os.path.splitext(file_name)[0]  # "00001"

        save_dir = os.path.join(OUTPUT_ROOT, subject_id)

        os.makedirs(save_dir, exist_ok=True)

        save_name = f"{folder}.png"  # "000-00.png"

        save_path = os.path.join(save_dir, save_name)

        # -----------------------------------------
        # SAVE
        # -----------------------------------------

        cv2.imwrite(save_path, img)

        gei_count += 1

        print(f"Saved: {save_path}")

# =====================================================
# DONE
# =====================================================

print("\n==========================")
print("Done")
print("Folders processed:", folder_count)
print("GEIs saved:", gei_count)