"""
Federated Learning trainer that orchestrates server and multiple clients.
"""

import os
import json
import shutil
import time
import torch
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.federated.privacy import DifferentialPrivacy
from src.data.loader import get_dataloader


# ---------------------------------------------------
# CLIENT PROCESS FUNCTION
# ---------------------------------------------------

def train_client_process(args):

    client_id, data_path, global_weights, device, local_epochs, batch_size, learning_rate = args

    from src.federated.client import FederatedClient

    client = FederatedClient(
        client_id=client_id,
        data_path=data_path,
        model=None,
        device=device,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    client.update_model(global_weights)

    weights, num_samples = client.train_local()

    return weights, num_samples


# ---------------------------------------------------
# FEDERATED TRAINER
# ---------------------------------------------------

class FederatedTrainer:

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        num_clients: int = 4,
        num_rounds: int = 2,
        local_epochs: int = 1,
        batch_size: int = 64,
        learning_rate: float = 2e-5,
        use_differential_privacy: bool = False,
        noise_multiplier: float = 0.01,
        max_grad_norm: float = 1.0,
        model_save_dir: str = "models/distilbert_federated",
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(model_save_dir, exist_ok=True)

        print(f"\nUsing device: {self.device}")

        self.server = FederatedServer(device=self.device)

        self.dp = (
            DifferentialPrivacy(noise_multiplier, max_grad_norm)
            if use_differential_privacy
            else None
        )

        self.client_data_paths = self._split_data_for_clients()

        self.history = {
            "rounds": [],
            "val_acc": [],
            "val_f1": []
        }


    # ---------------------------------------------------
    # DATA SPLITTING
    # ---------------------------------------------------

    def _split_data_for_clients(self):

        print("\nPreparing federated client datasets...")

        if os.path.exists("data/federated"):
            shutil.rmtree("data/federated")

        os.makedirs("data/federated", exist_ok=True)

        df = pd.read_csv(self.train_csv)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        chunk_size = len(df) // self.num_clients
        splits = []

        for i in range(self.num_clients):

            start = i * chunk_size
            end = start + chunk_size if i < self.num_clients - 1 else len(df)

            client_df = df.iloc[start:end]

            path = f"data/federated/client_{i}.csv"
            client_df.to_csv(path, index=False)

            splits.append(path)

            print(f"Client {i}: {len(client_df)} samples → {path}")

        return splits


    # ---------------------------------------------------
    # GLOBAL MODEL EVALUATION
    # ---------------------------------------------------

    def evaluate_global_model(self):

        loader = get_dataloader(
            self.val_csv,
            batch_size=64,
            shuffle=False
        )

        model = self.server.global_model.to(self.device)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for batch in loader:

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].cpu()

                logits = model(input_ids, attention_mask)

                preds = torch.argmax(logits, dim=1).cpu().tolist()

                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")

        return acc, f1


    # ---------------------------------------------------
    # FEDERATED TRAINING LOOP
    # ---------------------------------------------------

    def train(self):

        print("\nStarting Federated Training")
        print(f"Clients: {self.num_clients}")
        print(f"Rounds: {self.num_rounds}")
        print(f"Differential Privacy: {self.use_dp}")
        print("=" * 60)

        best_f1 = 0.0

        # Progress bar for rounds
        for round_num in tqdm(range(1, self.num_rounds + 1), desc="Federated Rounds"):

            start_time = time.time()

            global_weights = self.server.get_global_weights()

            client_updates = []

            # -----------------------------------------
            # PARALLEL CLIENT TRAINING
            # -----------------------------------------

            args_list = []

            for client_id, data_path in enumerate(self.client_data_paths):

                args = (
                    client_id,
                    data_path,
                    global_weights,
                    self.device,
                    self.local_epochs,
                    self.batch_size,
                    self.learning_rate
                )

                args_list.append(args)

            print("\nTraining clients in parallel...")

            # Progress bar for clients
            with mp.Pool(processes=2) as pool:

                results = list(
                    tqdm(
                        pool.imap(train_client_process, args_list),
                        total=len(args_list),
                        desc="Clients"
                    )
                )

            for weights, num_samples in results:

                if self.dp:
                    weights = self.dp.add_noise_to_weights(weights, num_samples)

                client_updates.append((weights, num_samples))

            # -----------------------------------------
            # SERVER AGGREGATION
            # -----------------------------------------

            self.server.aggregate(client_updates)

            acc, f1 = self.evaluate_global_model()

            print(f"\nGlobal Model — Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

            self.history["rounds"].append(round_num)
            self.history["val_acc"].append(acc)
            self.history["val_f1"].append(f1)

            if f1 > best_f1:

                best_f1 = f1

                self.server.save_model(
                    os.path.join(self.model_save_dir, "best_model.pt")
                )

                print(f"✓ Best federated model saved (F1={best_f1:.4f})")

            checkpoint_path = os.path.join(self.model_save_dir, f"round_{round_num}.pt")

            torch.save({
                "round": round_num,
                "model_state": self.server.global_model.state_dict(),
                "history": self.history
            }, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

            round_time = (time.time() - start_time) / 60
            print(f"Round time: {round_time:.2f} minutes")


        # ------------------------------------------------
        # SAVE FINAL MODEL
        # ------------------------------------------------

        self.server.save_model(
            os.path.join(self.model_save_dir, "final_model.pt")
        )

        with open(
            os.path.join(self.model_save_dir, "history.json"),
            "w"
        ) as f:
            json.dump(self.history, f, indent=2)

        if self.dp:
            self.dp.get_privacy_report(self.num_clients, self.num_rounds)

        print("\nFederated training complete")
        print(f"Best Val F1: {best_f1:.4f}")

        return self.history


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)