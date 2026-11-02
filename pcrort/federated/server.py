"""Production federated-server entry point; no local simulation exists here."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FederationEndpoint:
    address: str
    min_available_clients: int
    rounds: int
    tls_certificates: Optional[tuple[bytes, bytes, bytes]] = None

    def validate(self) -> None:
        if self.min_available_clients < 2:
            raise ValueError("Production federation requires at least two independent sites.")
        if self.rounds < 1:
            raise ValueError("rounds must be positive.")
        if not self.tls_certificates:
            raise ValueError("Refusing to start without mTLS certificates.")


def start_production_server(endpoint: FederationEndpoint) -> None:
    """Start a TLS-protected coordinator; it never reads patient-level data."""
    endpoint.validate()
    try:
        import flwr as fl
    except ImportError as exc:
        raise RuntimeError("Install the production dependency with `pip install flwr`.") from exc
    ca, cert, key = endpoint.tls_certificates
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=endpoint.min_available_clients,
        min_fit_clients=endpoint.min_available_clients,
        min_evaluate_clients=endpoint.min_available_clients,
    )
    fl.server.start_server(
        server_address=endpoint.address,
        config=fl.server.ServerConfig(num_rounds=endpoint.rounds),
        strategy=strategy,
        certificates=(ca, cert, key),
    )
