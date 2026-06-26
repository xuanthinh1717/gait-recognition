import pandas as pd

df = pd.read_csv("data/splits/train.csv")
print("Max label:", df["label"].max())
print("Min label:", df["label"].min())
print("Num unique:", df["label"].nunique())