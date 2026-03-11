import pandas as pd

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

print("Before dedup:")
print("Train:", len(train))
print("Val:", len(val))

train = train.drop_duplicates(subset=["text"])
val = val.drop_duplicates(subset=["text"])

print("\nAfter dedup:")
print("Train:", len(train))
print("Val:", len(val))

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)

print("\nDuplicates removed.")