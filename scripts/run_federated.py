"""
Run federated learning training.
Run: python scripts/run_federated.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.federated.federated_trainer import FederatedTrainer


def main():
    trainer = FederatedTrainer(
        train_csv='data/processed/train.csv',
        val_csv='data/processed/val.csv',
        num_clients=8,
        num_rounds=10,
        local_epochs=2,
        batch_size=16,
        learning_rate=2e-5,
        use_differential_privacy=True,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
        model_save_dir='models/distilbert_federated'
    )
    history = trainer.train()
    print("\nFederated training complete!")


if __name__ == '__main__':
    main()