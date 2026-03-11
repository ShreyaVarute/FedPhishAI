"""
Federated Learning trainer that orchestrates server and multiple clients.
"""
import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.federated.privacy import DifferentialPrivacy
from src.data.loader import get_dataloader


class FederatedTrainer:
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        num_clients: int = 3,
        num_rounds: int = 10,
        local_epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        use_differential_privacy: bool = False,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        model_save_dir: str = 'models/distilbert_federated',
        device: str = None
    ):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_dp = use_differential_privacy
        self.model_save_dir = model_save_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs('data/federated', exist_ok=True)

        self.server = FederatedServer(device=self.device)
        self.dp = DifferentialPrivacy(noise_multiplier, max_grad_norm) if use_differential_privacy else None

        # Split train data across clients
        self.client_data_paths = self._split_data_for_clients()

        self.history = {'rounds': [], 'val_acc': [], 'val_f1': []}

    def _split_data_for_clients(self):
        """Split training data into non-IID partitions for each client."""
        df = pd.read_csv(self.train_csv)
        # Simple IID split for now
        splits = []
        chunk_size = len(df) // self.num_clients
        for i in range(self.num_clients):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_clients - 1 else len(df)
            client_df = df.iloc[start:end]
            path = f'data/federated/client_{i}.csv'
            client_df.to_csv(path, index=False)
            splits.append(path)
            print(f"Client {i}: {len(client_df)} samples saved to {path}")
        return splits

    def evaluate_global_model(self):
        """Evaluate global model on validation set."""
        loader = get_dataloader(self.val_csv, batch_size=32, shuffle=False)
        model = self.server.global_model
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        return acc, f1

    def train(self):
        print(f"\nStarting Federated Training")
        print(f"Clients: {self.num_clients} | Rounds: {self.num_rounds} | DP: {self.use_dp}")
        print("=" * 60)

        best_f1 = 0.0

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n=== Federated Round {round_num}/{self.num_rounds} ===")
            global_weights = self.server.get_global_weights()
            client_updates = []

            # Each client trains locally
            for client_id, data_path in enumerate(self.client_data_paths):
                client = FederatedClient(
                    client_id=client_id,
                    data_path=data_path,
                    model=self.server.global_model,
                    device=self.device,
                    local_epochs=self.local_epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate
                )
                client.update_model(global_weights)
                weights, num_samples = client.train_local()

                # Apply differential privacy noise to weights
                if self.dp:
                    weights = self.dp.add_noise_to_weights(weights, num_samples)

                client_updates.append((weights, num_samples))

            # Server aggregates weights
            self.server.aggregate(client_updates)

            # Evaluate
            acc, f1 = self.evaluate_global_model()
            print(f"Global Model — Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

            self.history['rounds'].append(round_num)
            self.history['val_acc'].append(acc)
            self.history['val_f1'].append(f1)

            if f1 > best_f1:
                best_f1 = f1
                self.server.save_model(os.path.join(self.model_save_dir, 'best_model.pt'))
                print(f"  ✓ Best federated model saved (F1={best_f1:.4f})")

        # Save final model and history
        self.server.save_model(os.path.join(self.model_save_dir, 'final_model.pt'))
        with open(os.path.join(self.model_save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        if self.dp:
            self.dp.get_privacy_report(self.num_clients, self.num_rounds)

        print(f"\nFederated training complete. Best Val F1: {best_f1:.4f}")
        return self.history