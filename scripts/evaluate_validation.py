import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from src.detection.model import PhishingDetector
from src.data.loader import EmailDataset
from src.data.preprocessor import EmailPreprocessor


def evaluate():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "models/distilbert_baseline/best_model.pt"
    val_path = "data/processed/val.csv"

    if not os.path.exists(model_path):
        print("Model not found. Train the model first.")
        return

    print("Loading validation dataset...")

    dataset = EmailDataset(val_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f"Validation samples: {len(dataset)}")

    model = PhishingDetector(num_labels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)

    print("\n==============================")
    print("VALIDATION RESULTS")
    print("==============================")

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)


if __name__ == "__main__":
    evaluate()