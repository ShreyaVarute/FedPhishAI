import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from src.federated.aggregation import fedavg_plus_plus

class FedAvgPlusPlus(FedAvg):
    """Custom Flower strategy using adaptive confidence-weighted aggregation."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        fit_results = [
            (
                [np.array(v) for v in parameters_to_ndarrays(fit_res.parameters)],
                fit_res.num_examples,
                fit_res.metrics or {}
            )
            for _, fit_res in results
        ]
        aggregated = fedavg_plus_plus(fit_results)
        print(f'[Round {server_round}] Aggregated {len(results)} clients')
        return ndarrays_to_parameters(aggregated), {}