import flwr as fl
import torch
from collections import OrderedDict
from src.detection.model import PhishingDetector
from src.detection.trainer import EmailDataset, train_one_epoch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.federated.privacy import add_dp_noise
import numpy as np

class PhishingClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_path, val_path, dp_noise=0.01):
        self.client_id  = client_id
        self.dp_noise   = dp_noise
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model      = PhishingDetector().to(self.device)
        self.train_data = DataLoader(EmailDataset(train_path), batch_size=16, shuffle=True)
        self.val_data   = DataLoader(EmailDataset(val_path),   batch_size=32)
        self.confidence = 0.5

    def get_parameters(self, config):
        params = [v.cpu().numpy() for v in self.model.state_dict().values()]
        return add_dp_noise(params, noise_scale=self.dp_noise)

    def set_parameters(self, parameters):
        state = OrderedDict(
            {k: torch.tensor(v) for k, v in
             zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=len(self.train_data)
        )
        loss = train_one_epoch(self.model, self.train_data, optimizer, scheduler, self.device)
        print(f'[Client {self.client_id}] Loss: {loss:.4f}')
        return self.get_parameters(config), len(self.train_data.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, conf_sum = 0, 0, 0
        with torch.no_grad():
            for batch in self.val_data:
                ids   = batch['input_ids'].to(self.device)
                mask  = batch['attention_mask'].to(self.device)
                lbls  = batch['label'].to(self.device)
                logits, _ = self.model(ids, mask)
                probs     = torch.softmax(logits, dim=1)
                conf_sum += probs.max(dim=1).values.sum().item()
                correct  += (logits.argmax(dim=1) == lbls).sum().item()
                total    += lbls.size(0)
        self.confidence = conf_sum / total
        accuracy        = correct / total
        return float(1 - accuracy), total, {
            'accuracy':   accuracy,
            'confidence': self.confidence
        }