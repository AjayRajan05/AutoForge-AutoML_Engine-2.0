from ..tracking.storage import Storage
from datetime import datetime


class ExperimentLogger:
    def __init__(self):
        self.storage = Storage()
        self.run_id = self.storage.generate_run_id()
        self.record = {
            "run_id": self.run_id,
            "timestamp": str(datetime.now()),
            "params": {},
            "metrics": {},
            "model": None,
        }

    def log_params(self, params):
        self.record["params"] = params

    def log_metrics(self, metrics):
        self.record["metrics"] = metrics

    def log_model(self, model_name):
        self.record["model"] = model_name

    def save(self):
        self.storage.save(self.record)