import numpy as np
import os

LABELS_PATH = "data/npy/oumvlp_labels.npy"
OUTPUT_DIR  = "data/npy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

labels = np.load(LABELS_PATH)
n = len(labels)

indices = np.random.permutation(n)

train_end = int(0.8 * n)
val_end   = int(0.9 * n)

np.save(os.path.join(OUTPUT_DIR, "idx_train.npy"), indices[:train_end])
np.save(os.path.join(OUTPUT_DIR, "idx_val.npy"),   indices[train_end:val_end])
np.save(os.path.join(OUTPUT_DIR, "idx_test.npy"),  indices[val_end:])

print("Train:", train_end)
print("Val:",   val_end - train_end)
print("Test:",  n - val_end)