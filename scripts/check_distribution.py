import pandas as pd

# Paths
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH   = "data/processed/val.csv"

def check_distribution(path, name):
    df = pd.read_csv(path)

    total = len(df)
    phishing = (df['label'] == 1).sum()
    legit = (df['label'] == 0).sum()

    print(f"\n{name} DATA")
    print("-" * 40)
    print(f"Total samples: {total}")
    print(f"Phishing (1): {phishing}")
    print(f"Legitimate (0): {legit}")

    print("\nProportion:")
    print(f"Phishing: {phishing/total:.4f}")
    print(f"Legitimate: {legit/total:.4f}")


if __name__ == "__main__":
    check_distribution(TRAIN_PATH, "TRAIN")
    check_distribution(VAL_PATH, "VALIDATION")