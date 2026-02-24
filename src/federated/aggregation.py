import numpy as np
from typing import List, Tuple

def fedavg_plus_plus(
    results: List[Tuple[List[np.ndarray], int, dict]]
) -> List[np.ndarray]:
    """
    FedAvg++ — Adaptive Confidence-Weighted Aggregation.

    Weight formula:
        client_weight = data_size × avg_confidence × stability_factor

    Benefits:
    - Reduces influence of noisy / low-quality clients
    - Improves robustness to skewed phishing ratios (non-IID)
    - Adds algorithmic novelty for the research paper
    """
    weights, param_list = [], []

    for params, data_size, metrics in results:
        confidence = metrics.get('confidence', 0.5)
        # Penalise over-confidence AND under-confidence
        stability  = 1.0 - abs(confidence - 0.75)
        weight     = data_size * confidence * max(stability, 0.1)
        weights.append(weight)
        param_list.append(params)

    total   = sum(weights)
    weights = [w / total for w in weights]  # normalise

    aggregated = [
        sum(w * p for w, p in zip(weights, [pl[i] for pl in param_list]))
        for i in range(len(param_list[0]))
    ]
    return aggregated