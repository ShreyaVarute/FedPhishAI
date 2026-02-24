import pandas as pd
import os

path = "data/raw/phishing"
dfs = []

for file in os.listdir(path):
    if file.endswith(".csv") and file != "all_phishing.csv":
        print("Reading:", file)

        df = pd.read_csv(os.path.join(path, file), encoding='latin1', on_bad_lines='skip')

        if 'body' in df.columns:
            df = df[['body']].rename(columns={'body': 'text'})
        elif 'text' in df.columns:
            df = df[['text']]
        elif 'text_combined' in df.columns:
            df = df[['text_combined']].rename(columns={'text_combined': 'text'})
        else:
            print("Skipping:", file)
            continue

        dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined['label'] = 1

os.makedirs("data/raw/phishing", exist_ok=True)
combined.to_csv("data/raw/phishing/all_phishing.csv", index=False)

print("All phishing files combined successfully")
print("Total rows:", len(combined))