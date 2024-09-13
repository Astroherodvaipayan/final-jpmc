import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict
from flwr.common import Metrics, FitRes, EvaluateRes, Parameters, NDArrays
from flwr.server.client_proxy import ClientProxy

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Metrics]]:
        self.current_round = rnd
        print(f"\n========================== ROUND {rnd} ==========================")
        
        if not results:
            return None, {}

        # Convert the list of results to weights and num_examples
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Print received weights from each client
        for idx, (weights, num_examples) in enumerate(weights_results):
            print(f"Received weights from client {idx}:")
            for i, layer in enumerate(weights):
                print(f"Layer {i}: {layer}")

        # Perform weighted averaging
        total_examples = sum(num_examples for _, num_examples in weights_results)
        aggregated_weights = [
            np.sum([
                weights[i] * num_examples for weights, num_examples in weights_results
            ], axis=0) / total_examples
            for i in range(len(weights_results[0][0]))
        ]

        # Print aggregated weights
        print(f"Aggregated weights for round {rnd}:")
        for i, layer in enumerate(aggregated_weights):
            print(f"Layer {i}: {layer}")

        # Return aggregated weights and an empty metrics dict
        return fl.common.ndarrays_to_parameters(aggregated_weights), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Metrics]]:
        if not results:
            return None, {}
        loss_aggregated = sum(r.loss * r.num_examples for _, r in results) / sum(r.num_examples for _, r in results)
        accuracy_aggregated = sum(r.metrics["accuracy"] * r.num_examples for _, r in results) / sum(r.num_examples for _, r in results)
        print(f"Round {rnd} performance metrics:")
        print(f"Aggregated loss: {loss_aggregated:.4f}")
        print(f"Aggregated accuracy: {accuracy_aggregated:.4f}")
        return loss_aggregated, {"accuracy": accuracy_aggregated}

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = fl.common.ndarrays_to_parameters([
            np.zeros((4,), dtype=np.float32),  # weights
            np.array([0.0], dtype=np.float32)  # bias
        ])
        return initial_parameters

strategy = CustomFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,  # Minimum 2 clients required for fitting
    min_evaluate_clients=2,  # Minimum 2 clients required for evaluation
    min_available_clients=2,  # Minimum 2 clients must be available before starting
    initial_parameters=None,  # This will use the initialize_parameters method
)

def run_server():
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    run_server()
