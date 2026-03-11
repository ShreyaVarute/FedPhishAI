import os
import json
import torch
import torch.nn as nn
import numpy as np

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from src.detection.model import PhishingDetector
from src.data.loader import get_dataloader


class Trainer:

    def __init__(
        self,
        train_csv,
        val_csv,
        model_save_dir,
        epochs=5,
        batch_size=32,
        learning_rate=2e-5,
        max_length=256,
        device=None
    ):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)

        os.makedirs(model_save_dir, exist_ok=True)

        self.model = PhishingDetector(num_labels=2).to(self.device)

        self.train_loader = get_dataloader(train_csv, batch_size=batch_size, shuffle=True, max_length=max_length)
        self.val_loader = get_dataloader(val_csv, batch_size=batch_size, shuffle=False, max_length=max_length)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # -------- CLASS WEIGHTS --------

        labels = self.train_loader.dataset.df['label'].values

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )

        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        print("Class weights:", class_weights)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # --------------------------------

        self.epochs = epochs
        self.model_save_dir = model_save_dir

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': []
        }

    def train_epoch(self):

        self.model.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.train_loader, desc="Training"):

            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(ids, mask)

            loss = self.criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(self.train_loader)
        acc = accuracy_score(all_labels, all_preds)

        return avg_loss, acc

    def evaluate(self):

        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():

            for batch in tqdm(self.val_loader, desc="Evaluating"):

                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(ids, mask)

                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)

        print("Prediction distribution:", {0: all_preds.count(0), 1: all_preds.count(1)})

        return avg_loss, acc, f1, precision, recall

    def train(self):

        best_f1 = 0

        for epoch in range(self.epochs):

            print(f"\nEpoch {epoch+1}/{self.epochs}")

            train_loss, train_acc = self.train_epoch()

            val_loss, val_acc, val_f1, val_precision, val_recall = self.evaluate()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            if val_f1 > best_f1:

                best_f1 = val_f1

                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_save_dir, "best_model.pt")
                )

                print("Best model saved")

        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_save_dir, "final_model.pt")
        )

        with open(os.path.join(self.model_save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        print("\nTraining finished")
        print("Best F1:", best_f1)

        return self.history