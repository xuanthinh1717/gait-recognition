import os
import cv2
import numpy as np

# =====================================================
# PATHS
# =====================================================

INPUT_ROOT = r"data/raw/CASIA-B"

OUTPUT_ROOT = r"data/processed/gei"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# PROCESS DATASET
# =====================================================

subject_count = 0
gei_count = 0

for subject in sorted(os.listdir(INPUT_ROOT)):

    subject_path = os.path.join(INPUT_ROOT, subject)

    if not os.path.isdir(subject_path):
        continue

    print(f"\nSubject: {subject}")

    subject_count += 1

    # -------------------------------------------------

    for sequence in sorted(os.listdir(subject_path)):

        sequence_path = os.path.join(subject_path, sequence)

        if not os.path.isdir(sequence_path):
            continue

        # ---------------------------------------------
        # ANGLES
        # ---------------------------------------------

        for angle in sorted(os.listdir(sequence_path)):

            angle_path = os.path.join(sequence_path, angle)

            if not os.path.isdir(angle_path):
                continue

            frames = []

            # -----------------------------------------
            # LOAD FRAMES
            # -----------------------------------------

            for file_name in sorted(os.listdir(angle_path)):

                if not file_name.endswith(".png"):
                    continue

                file_path = os.path.join(angle_path, file_name)

                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img = img.astype("float32") / 255.0

                frames.append(img)

            # -----------------------------------------
            # CHECK
            # -----------------------------------------

            if len(frames) == 0:
                continue

            # -----------------------------------------
            # GENERATE GEI
            # -----------------------------------------

            gei = np.mean(frames, axis=0)

            # -----------------------------------------
            # OUTPUT PATH
            # -----------------------------------------

            save_dir = os.path.join(
                OUTPUT_ROOT,
                subject
            )

            os.makedirs(save_dir, exist_ok=True)

            save_name = f"{sequence}_{angle}.png"

            save_path = os.path.join(save_dir, save_name)

            # -----------------------------------------
            # SAVE
            # -----------------------------------------

            cv2.imwrite(
                save_path,
                (gei * 255).astype("uint8")
            )

            gei_count += 1

            print(f"Saved: {save_path}")

# =====================================================
# DONE
# =====================================================

print("\n==========================")
print("Done")
print("Subjects:", subject_count)
print("GEIs:", gei_count)