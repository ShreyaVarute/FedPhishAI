"""
Centralized DistilBERT training (with history saving + resume support)

Run:
python -m scripts.run_baseline
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.detection.model import PhishingDetector
from src.detection.trainer import EmailDataset, train_one_epoch
from src.detection.evaluator import evaluate_model

# ---------------- CONFIG ---------------- #
EPOCHS = 2
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 32

TRAIN = 'data/processed/train.csv'
VAL   = 'data/processed/val.csv'

SAVE_DIR = 'models/distilbert_baseline'
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(SAVE_DIR, "best_model.pt")
HISTORY_PATH = os.path.join(SAVE_DIR, "history.json")

# ---------------- DEVICE ---------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ---------------- MODEL ---------------- #
model = PhishingDetector().to(device)

# Resume if model exists
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model.load_state_dict(torch.load(MODEL_PATH))

# ---------------- DATA ---------------- #
train_loader = DataLoader(
    EmailDataset(TRAIN),
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=0,   # IMPORTANT: Windows fix
)

val_loader = DataLoader(
    EmailDataset(VAL),
    batch_size=BATCH_SIZE_VAL,
    num_workers=0,
)

# ---------------- OPTIMIZER ---------------- #
optimizer = AdamW(model.parameters(), lr=2e-5)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCHS * len(train_loader)
)

# ---------------- HISTORY ---------------- #
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)
else:
    history = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

best_f1 = max(history["f1"]) if history["f1"] else 0.0

# ---------------- TRAIN LOOP ---------------- #
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
    metrics = evaluate_model(model, val_loader, device)

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")

    # Save history
    history["loss"].append(loss)
    history["accuracy"].append(metrics["accuracy"])
    history["precision"].append(metrics["precision"])
    history["recall"].append(metrics["recall"])
    history["f1"].append(metrics["f1"])

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    # Save best model
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        torch.save(model.state_dict(), MODEL_PATH)
        print("Best model saved")

# Save final model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pt"))

print("\nTraining complete")
print(f"Best F1: {best_f1:.4f}")