print("START")

import os
import cv2
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

INPUT_IMAGE = "data/raw/CASIA-B/001/bg-01/000/001-bg-01-000-001.png"
OUTPUT_DIR = "data/processed"

# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD IMAGE
# =====================================================

img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Failed to load image")
    exit()

print("Original shape:", img.shape)

# =====================================================
# RESIZE
# =====================================================

img_resized = cv2.resize(img, (64, 64))

# =====================================================
# NORMALIZE
# =====================================================

img_normalized = img_resized / 255.0

# =====================================================
# SAVE
# =====================================================

output_path = os.path.join(OUTPUT_DIR, "processed.png")

cv2.imwrite(output_path, (img_normalized * 255).astype("uint8"))

print("Saved to:", output_path)

# =====================================================
# SHOW IMAGE
# =====================================================

plt.imshow(img_normalized, cmap="gray")
plt.title("Processed Image")
plt.axis("off")
plt.show()