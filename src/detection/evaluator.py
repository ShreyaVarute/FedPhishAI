import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import numpy as np
from src.detection.model import PhishingDetector
from src.data.loader import get_dataloader


class Evaluator:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PhishingDetector(num_labels=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def evaluate(self, csv_path: str, batch_size: int = 32):
        loader = get_dataloader(csv_path, batch_size=batch_size, shuffle=False)
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']

                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(probs[:, 1].cpu().tolist())  # phishing probability

        print("\n=== Evaluation Results ===")
        print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
        print(f"F1 Score:  {f1_score(all_labels, all_preds, average='binary'):.4f}")
        print(f"Precision: {precision_score(all_labels, all_preds, average='binary', zero_division=0):.4f}")
        print(f"Recall:    {recall_score(all_labels, all_preds, average='binary', zero_division=0):.4f}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")
        print(f"\nClassification Report:\n{classification_report(all_labels, all_preds, target_names=['Legitimate', 'Phishing'])}")

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='binary'),
            'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }