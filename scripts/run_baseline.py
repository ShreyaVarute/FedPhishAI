"""
Main script to train the DistilBERT phishing detector baseline.
Run: python scripts/run_baseline.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.trainer import Trainer

def main():
    trainer = Trainer(
        train_csv='data/processed/train.csv',
        val_csv='data/processed/val.csv',
        model_save_dir='models/distilbert_baseline',
        epochs=3,
        batch_size=32,
        learning_rate=2e-5,
        max_length=256
    )
    history = trainer.train()
    print("\nFinal training history saved to models/distilbert_baseline/history.json")

if __name__ == '__main__':
    main()