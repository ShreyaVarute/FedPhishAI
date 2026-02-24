import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_all

def check_full_distribution(path, name):
    df = load_all()

    total = len(df)
    phishing = (df['label'] == 1).sum()
    legit = (df['label'] == 0).sum()

    print("FULL DATASET")
    print("-" * 40)
    print("Total samples:", total)
    print("Phishing (1):", phishing)
    print("Legitimate (0):", legit)

    print("\nProportion:")
    print("Phishing:", phishing / total)
    print("Legitimate:", legit / total)

if __name__ == "__main__":
    check_full_distribution(TRAIN_PATH, "TRAIN")
    check_full_distribution(VAL_PATH, "VALIDATION")