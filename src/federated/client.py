import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.loader import get_dataloader


class FederatedClient:

    def __init__(
        self,
        client_id: int,
        data_path: str,
        model,
        device: str,
        local_epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):

        self.client_id = client_id
        self.data_path = data_path
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if model is None:
            raise ValueError("Model cannot be None for FederatedClient")

        # Local copy of global model
        self.model = copy.deepcopy(model).to(device)

        self.criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------
    # UPDATE LOCAL MODEL
    # ---------------------------------------------------

    def update_model(self, global_weights):
        """Update local model with global weights from server."""
        self.model.load_state_dict(copy.deepcopy(global_weights))

    # ---------------------------------------------------
    # LOCAL TRAINING
    # ---------------------------------------------------

    def train_local(self):

        self.model.train()

        loader = get_dataloader(
            self.data_path,
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        total_steps = len(loader) * self.local_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps
        )

        total_loss = 0

        for epoch in range(self.local_epochs):

            print(f"\nClient {self.client_id} | Epoch {epoch+1}/{self.local_epochs}")

            progress = tqdm(loader, desc=f"Client {self.client_id} Training")

            for batch in progress:

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                logits = self.model(input_ids, attention_mask)

                loss = self.criterion(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / (len(loader) * self.local_epochs)

        print(f"Client {self.client_id} finished | Avg loss = {avg_loss:.4f}")

        return self.model.state_dict(), len(loader.dataset)