import torch
import numpy as np
import copy


class DifferentialPrivacy:
    """
    Differential Privacy for Federated Learning using Gaussian mechanism.
    Clips gradients and adds calibrated noise to protect client data.
    """

    def __init__(self, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        """
        Args:
            noise_multiplier: Controls noise level (higher = more privacy, less accuracy)
            max_grad_norm: Gradient clipping norm (sensitivity)
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def clip_gradients(self, model):
        """Clip gradients to bound sensitivity."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)

    def add_noise_to_weights(self, state_dict: dict, num_samples: int) -> dict:
        """Add Gaussian noise to model weights for differential privacy."""
        noisy_state = copy.deepcopy(state_dict)
        sigma = self.noise_multiplier * self.max_grad_norm / num_samples

        for key in noisy_state:
            if noisy_state[key].dtype in [torch.float32, torch.float64]:
                noise = torch.normal(
                    mean=0.0,
                    std=sigma,
                    size=noisy_state[key].shape
                ).to(noisy_state[key].device)
                noisy_state[key] = noisy_state[key] + noise

        return noisy_state

    def compute_epsilon(self, num_steps: int, delta: float = 1e-5) -> float:
        """
        Approximate privacy budget epsilon using the moments accountant.
        Simplified estimation.
        """
        # Simple approximation using RDP accountant
        q = 1.0  # sampling rate (full batch)
        alpha = 10  # order
        # RDP epsilon for Gaussian mechanism
        rdp_epsilon = alpha * (self.noise_multiplier ** -2) / 2 * num_steps
        # Convert RDP to (epsilon, delta)-DP
        epsilon = rdp_epsilon + np.log(1 / delta) / (alpha - 1)
        return float(epsilon)

    def get_privacy_report(self, num_clients: int, num_rounds: int, delta: float = 1e-5):
        """Generate a privacy budget report."""
        epsilon = self.compute_epsilon(num_rounds * num_clients, delta)
        print(f"\n=== Differential Privacy Report ===")
        print(f"Noise multiplier (σ): {self.noise_multiplier}")
        print(f"Gradient clipping norm: {self.max_grad_norm}")
        print(f"Clients: {num_clients} | Rounds: {num_rounds}")
        print(f"Privacy budget: ε ≈ {epsilon:.4f}, δ = {delta}")
        print(f"Interpretation: {'Strong privacy' if epsilon < 1 else 'Moderate privacy' if epsilon < 10 else 'Weak privacy'}")
        return epsilon