import pandas as pd

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

print("Train size:", len(train))
print("Validation size:", len(val))

# check overlap
common = pd.merge(train, val)

print("Common rows between train and val:", len(common))