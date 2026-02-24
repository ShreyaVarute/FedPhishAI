import numpy as np
from typing import List

def add_dp_noise(
    parameters: List[np.ndarray],
    noise_scale: float = 0.01,
    clip_norm:   float = 1.0
) -> List[np.ndarray]:
    """
    Gaussian Mechanism for Differential Privacy.
    1. Clip gradients to L2 norm bound (bounding sensitivity)
    2. Add Gaussian noise scaled to the sensitivity
    Set noise_scale=0.0 to disable DP entirely.
    """
    noisy_params = []
    for param in parameters:
        norm = np.linalg.norm(param)
        if norm > clip_norm:
            param = param * (clip_norm / norm)  # gradient clipping
        noise = np.random.normal(0, noise_scale, param.shape).astype(param.dtype)
        noisy_params.append(param + noise)
    return noisy_params