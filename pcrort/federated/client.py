"""Production federated client adapter; deploy one at each data-owning site."""
from __future__ import annotations
from typing import Callable, Dict, List
import numpy as np
import torch


class PCRORTClient:
    """Uses only a site-local trainer and rejects incompatible model payloads."""
    def __init__(self, trainer_factory: Callable[[], object]):
        self.trainer = trainer_factory()
        self.model, self.device = self.trainer.model, self.trainer.device

    def get_parameters(self, config: Dict | None = None) -> List[np.ndarray]:
        return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state = self.model.state_dict()
        if len(parameters) != len(state):
            raise ValueError("Rejected incompatible parameter payload.")
        for key, value in zip(state, parameters):
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"Rejected incompatible tensor for {key}.")
            state[key] = torch.as_tensor(value, device=self.device, dtype=state[key].dtype)
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict):
        self.set_parameters(parameters)
        for epoch in range(int(config.get("local_epochs", 1))):
            metrics = self.trainer.train_epoch(epoch)
        return self.get_parameters(), len(self.trainer.train_loader.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict):
        self.set_parameters(parameters)
        metrics, _ = self.trainer.evaluate(self.trainer.val_loader, prefix="val")
        return float(metrics.get("val_loss", 0.0)), len(self.trainer.val_loader.dataset), metrics


def start_production_client(server_address: str, client: PCRORTClient, root_certificates: bytes) -> None:
    try:
        import flwr as fl
    except ImportError as exc:
        raise RuntimeError("Install the production dependency with `pip install flwr`.") from exc
    class FlowerAdapter(fl.client.NumPyClient):
        def get_parameters(self, config):
            return client.get_parameters(config)

        def fit(self, parameters, config):
            return client.fit(parameters, config)

        def evaluate(self, parameters, config):
            return client.evaluate(parameters, config)

    fl.client.start_numpy_client(
        server_address=server_address,
        client=FlowerAdapter(),
        root_certificates=root_certificates,
    )
