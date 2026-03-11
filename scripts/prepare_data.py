import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
import re


def clean_email_text(text: str):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def load_enron(path):
    df = pd.read_csv(path)

    if 'message' in df.columns:
        df = df.rename(columns={'message': 'text'})
    elif 'body' in df.columns:
        df = df.rename(columns={'body': 'text'})

    df['label'] = 0
    df = df[['text', 'label']].dropna()

    print(f"Enron: {len(df)} legitimate emails loaded")
    return df


def load_phishing_csv(path):
    df = pd.read_csv(path)

    text_col = None
    label_col = None

    for col in df.columns:
        if text_col is None and ("text" in col.lower() or "email" in col.lower() or "body" in col.lower()):
            text_col = col

        if label_col is None and ("label" in col.lower() or "type" in col.lower() or "class" in col.lower()):
            label_col = col

    df = df.rename(columns={text_col: 'text', label_col: 'label'})

    if df['label'].dtype == object:
        df['label'] = df['label'].apply(
            lambda x: 1 if str(x).lower() in ['phishing', 'spam', 'phishing email', '1'] else 0
        )
    else:
        df['label'] = df['label'].astype(int)

    df = df[['text', 'label']].dropna()

    print(
        f"Loaded {os.path.basename(path)} | "
        f"Total={len(df)} | "
        f"Phishing={df['label'].sum()} | "
        f"Legit={(df['label']==0).sum()}"
    )

    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=['text'])
    after = len(df)

    print(f"Removed {before-after} duplicates. Remaining: {after}")
    return df

def main():

    os.makedirs("data/processed", exist_ok=True)

    dfs = []

    phishing_files = [
        "data/raw/phishing/all_phishing.csv",
        "data/raw/phishing/CEAS_08.csv",
        "data/raw/phishing/Enron.csv",
        "data/raw/phishing/Ling.csv",
        "data/raw/phishing/Nazario.csv",
        "data/raw/phishing/Nigerian_Fraud.csv",
        "data/raw/phishing/phishing_email.csv",
        "data/raw/phishing/SpamAssasin.csv"
    ]

    for path in phishing_files:
        if os.path.exists(path):
            dfs.append(load_phishing_csv(path))

    enron_path = "data/raw/enron/emails.csv"

    if os.path.exists(enron_path):
        dfs.append(load_enron(enron_path))

    combined = pd.concat(dfs, ignore_index=True)

    print("\nTotal combined:", len(combined))
    print("Phishing:", combined['label'].sum(),
          "| Legitimate:", (combined['label']==0).sum())

    combined['text'] = combined['text'].apply(clean_email_text)

    combined = combined[combined['text'].str.len() > 50]

    combined = remove_duplicates(combined)
    
    # -------- SAMPLE DATASET (for fast testing) --------
    USE_SAMPLE = True
    SAMPLE_SIZE = 50000

    if USE_SAMPLE:
        combined = combined.sample(
            n=min(SAMPLE_SIZE, len(combined)),
            random_state=42
        )
        print(f"\nUsing SAMPLE dataset: {len(combined)} emails")
#------------------------------------------------------------------------
    print("\nUsing FULL dataset (class weights will handle imbalance)")

    train_df, val_df = train_test_split(
        combined,
        test_size=0.2,
        random_state=42,
        stratify=combined['label']
    )

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    combined.to_csv("data/processed/all.csv", index=False)

    print("\nDataset saved:")
    print("Train:", len(train_df))
    print("Val:", len(val_df))

    print("\nTrain distribution:")
    print(train_df['label'].value_counts())

    print("\nValidation distribution:")
    print(val_df['label'].value_counts())


if __name__ == "__main__":
    main()