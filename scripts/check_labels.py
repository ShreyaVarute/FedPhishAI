import pandas as pd

train = pd.read_csv("data/processed/train.csv")

print(train.head())

print("\nLabel distribution:")
print(train["label"].value_counts())
print("\nProportion:")
print(train["label"].value_counts(normalize=True))