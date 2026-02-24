"""
Privacy-Utility Tradeoff Experiment.
Tests accuracy at different DP noise levels.
Run: python scripts/run_privacy_tradeoff.py
"""
import json, os, flwr as fl
from src.federated.client import PhishingClient
from src.federated.server import FedAvgPlusPlus

NOISE_LEVELS = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5]
NUM_CLIENTS  = 8
NUM_ROUNDS   = 5   # Fewer rounds per experiment for speed
results      = []

for noise in NOISE_LEVELS:
    print(f'\n=== DP Noise = {noise} ===')

    history = fl.simulation.start_simulation(
        client_fn  = lambda cid: PhishingClient(
            int(cid),
            f'data/federated/client_{cid}/train.csv',
            f'data/federated/client_{cid}/val.csv',
            dp_noise=noise
        ),
        num_clients      = NUM_CLIENTS,
        config           = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy         = FedAvgPlusPlus(
            min_available_clients=NUM_CLIENTS,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
        ),
        client_resources = {'num_cpus': 1},
    )
    # Get accuracy from last round
    last_acc = list(history.metrics_distributed.get('accuracy', {}).values())[-1]
    results.append({'noise': noise, 'accuracy': last_acc})
    print(f'Noise: {noise} → Accuracy: {last_acc:.4f}')

os.makedirs('experiments/results', exist_ok=True)
with open('experiments/results/privacy_tradeoff.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to experiments/results/privacy_tradeoff.json')
