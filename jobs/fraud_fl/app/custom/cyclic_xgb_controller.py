"""Cyclic controller for XGBoost federated learning.

Passes the model serially from client to client. Each client warm-starts
from the received model, trains locally, and sends the updated model to
the next client. This avoids the need for numerical parameter averaging
which doesn't work with serialised XGBoost boosters.
"""
import json
from pathlib import Path

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.model_controller import ModelController


class CyclicXGBController(ModelController):
    def __init__(
        self,
        num_rounds: int = 10,
        task_name: str = "train",
        save_filename: str = "global_fraud_model.json",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.task_name = task_name
        self.save_filename = save_filename

    def run(self) -> None:
        self.info("Starting Cyclic XGBoost training")

        clients = self.sample_clients(0)
        self.info(f"Participating clients: {clients}")

        model = FLModel(params={}, params_type="FULL")

        for rnd in range(self.num_rounds):
            self.info(f"--- Round {rnd + 1}/{self.num_rounds} ---")
            for client in clients:
                client_name = client if isinstance(client, str) else client.name
                self.info(f"  Sending model to {client_name}")
                results = self.send_model_and_wait(
                    task_name=self.task_name,
                    data=model,
                    targets=[client],
                    timeout=600,
                )
                if results and len(results) > 0:
                    model = results[0]
                    self.info(f"  Received updated model from {client_name}")
                else:
                    self.warning(f"  No result from {client_name}")

        # Save the final global model
        if model and model.params and model.params.get("model_bytes"):
            app_dir = self.fl_ctx.get_prop("APP_ROOT", ".")
            save_path = Path(app_dir) / self.save_filename
            save_path.write_bytes(model.params["model_bytes"])
            self.info(f"Saved global model to {save_path}")

        self.info("Cyclic XGBoost training complete")
