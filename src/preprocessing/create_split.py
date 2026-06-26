import os
import random
import pandas as pd

# =====================================================
# CONFIG
# =====================================================

GEI_ROOT = r"data/processed/gei_oumvlp"

OUTPUT_DIR = r"data/splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# COLLECT DATA
# =====================================================

rows = []

for subject in sorted(os.listdir(GEI_ROOT)):

    subject_path = os.path.join(GEI_ROOT, subject)

    if not os.path.isdir(subject_path):
        continue

    for file_name in os.listdir(subject_path):

        if not file_name.endswith(".png"):
            continue

        path = os.path.join(subject_path, file_name)

        rows.append({
            "path": path,
            "label": int(subject) - 1
        })

# =====================================================
# SHUFFLE
# =====================================================

random.shuffle(rows)

# =====================================================
# SPLIT
# =====================================================

train_size = int(0.8 * len(rows))
val_size = int(0.1 * len(rows))

train_rows = rows[:train_size]
val_rows = rows[train_size:train_size + val_size]
test_rows = rows[train_size + val_size:]

# =====================================================
# SAVE CSV
# =====================================================

pd.DataFrame(train_rows).to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

pd.DataFrame(val_rows).to_csv(
    os.path.join(OUTPUT_DIR, "val.csv"),
    index=False
)

pd.DataFrame(test_rows).to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

# =====================================================
# INFO
# =====================================================

print("Train:", len(train_rows))
print("Val:", len(val_rows))
print("Test:", len(test_rows))