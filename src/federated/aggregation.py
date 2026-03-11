import torch
import copy


def fedavg(client_updates: list) -> dict:
    """
    Federated Averaging (FedAvg) algorithm.
    Aggregates model weights from multiple clients using weighted average.

    Args:
        client_updates: list of (state_dict, num_samples) tuples

    Returns:
        Aggregated state dict
    """
    total_samples = sum(n for _, n in client_updates)
    aggregated = copy.deepcopy(client_updates[0][0])

    for key in aggregated.keys():
        aggregated[key] = torch.zeros_like(aggregated[key], dtype=torch.float32)
        for state_dict, num_samples in client_updates:
            weight = num_samples / total_samples
            aggregated[key] += state_dict[key].float() * weight

    return aggregated


def fedprox(client_updates: list, global_weights: dict, mu: float = 0.01) -> dict:
    """
    FedProx aggregation with proximal term.
    Similar to FedAvg but penalizes deviation from global model.
    """
    # For aggregation, FedProx uses same averaging as FedAvg
    # The proximal term is applied during client training
    return fedavg(client_updates)