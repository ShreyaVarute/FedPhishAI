import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_all
from sklearn.model_selection import train_test_split

def balance_dataset(df):
    phishing = df[df.label == 1]
    legit    = df[df.label == 0]

    min_size = min(len(phishing), len(legit))

    phishing = phishing.sample(min_size, random_state=42)
    legit    = legit.sample(min_size, random_state=42)

    return pd.concat([phishing, legit]).sample(frac=1, random_state=42)

import pandas as pd

# Load full dataset
df = load_all()

# Balance it
df = balance_dataset(df)

# Sample (VERY IMPORTANT for speed)
df = df.sample(n=100000, random_state=42)

print("Total data:", len(df))

# Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs("data/processed", exist_ok=True)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Data preparation complete")