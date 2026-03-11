import pandas as pd

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

overlap = set(train["text"]).intersection(set(val["text"]))

print("Train size:", len(train))
print("Validation size:", len(val))
print("Common emails:", len(overlap))