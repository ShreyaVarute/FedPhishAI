"""
Federated training simulation (all clients in-process via Flower).
Run: python scripts/run_federated.py
"""
import flwr as fl
from src.federated.client import PhishingClient
from src.federated.server import FedAvgPlusPlus

NUM_CLIENTS = 8
NUM_ROUNDS  = 10
DP_NOISE    = 0.01   # Change for privacy tradeoff experiments

def client_fn(cid: str):
    i = int(cid)
    return PhishingClient(
        client_id  = i,
        train_path = f'data/federated/client_{i}/train.csv',
        val_path   = f'data/federated/client_{i}/val.csv',
        dp_noise   = DP_NOISE
    )

strategy = FedAvgPlusPlus(
    min_available_clients = NUM_CLIENTS,
    min_fit_clients       = NUM_CLIENTS,
    min_evaluate_clients  = NUM_CLIENTS,
)

fl.simulation.start_simulation(
    client_fn        = client_fn,
    num_clients      = NUM_CLIENTS,
    config           = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy         = strategy,
    client_resources = {'num_cpus': 1},
)
print('Federated training complete.')