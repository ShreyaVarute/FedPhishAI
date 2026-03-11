import copy
import torch
from src.detection.model import PhishingDetector


class FederatedServer:
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = PhishingDetector(num_labels=2).to(self.device)

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates: list):
        """
        FedAvg aggregation: weighted average of client model weights.
        client_updates: list of (state_dict, num_samples) tuples
        """
        total_samples = sum(n for _, n in client_updates)
        aggregated = copy.deepcopy(client_updates[0][0])

        for key in aggregated.keys():
            aggregated[key] = torch.zeros_like(aggregated[key], dtype=torch.float32)
            for state_dict, num_samples in client_updates:
                weight = num_samples / total_samples
                aggregated[key] += state_dict[key].float() * weight

        self.global_model.load_state_dict(aggregated)
        print(f"  Server: aggregated weights from {len(client_updates)} clients ({total_samples} total samples)")

    def save_model(self, path: str):
        torch.save(self.global_model.state_dict(), path)
        print(f"Global model saved to {path}")

    def load_model(self, path: str):
        self.global_model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Global model loaded from {path}")