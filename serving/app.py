"""REST inference API for trained AutoForge models."""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    BaseModel = object

try:
    from core.unified_automl import UnifiedAutoML
except ImportError:
    from ..core.unified_automl import UnifiedAutoML

try:
    from systemization.systemization_integration import systemization_integrator
except ImportError:
    try:
        from ..systemization.systemization_integration import systemization_integrator
    except ImportError:
        systemization_integrator = None


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    ground_truth: Optional[List[Any]] = None


class FeedbackRequest(BaseModel):
    prediction_id: str
    y_true: Any


def create_app(model_path: Optional[str] = None) -> "FastAPI":
    if not FASTAPI_AVAILABLE:
        raise ImportError("Install serving extras: pip install 'autoforge[serve]'")

    app = FastAPI(title="AutoForge Serving API", version="1.0.0")
    automl = UnifiedAutoML()
    loaded_path: Optional[str] = None
    monitor = None
    prediction_log: Dict[str, Dict[str, Any]] = {}

    if systemization_integrator:
        monitor = systemization_integrator.setup_monitoring(automl)

    if model_path:
        automl.load_model(model_path)
        loaded_path = model_path

    @app.on_event("startup")
    def startup_monitor():
        nonlocal monitor
        if monitor is None and systemization_integrator:
            monitor = systemization_integrator.setup_monitoring(automl)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": automl.is_fitted,
            "model_path": loaded_path,
            "monitoring": monitor is not None,
        }

    @app.post("/load/{model_name}")
    def load_model(model_name: str):
        nonlocal loaded_path
        try:
            automl.load_model(model_name)
            loaded_path = model_name
            return {"status": "loaded", "model": model_name}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/predict")
    def predict(request: PredictRequest):
        if not automl.is_fitted:
            raise HTTPException(status_code=400, detail="No model loaded")
        try:
            start = time.time()
            df = pd.DataFrame(request.records)
            preds = automl.predict(df)
            latency = time.time() - start
            prediction_id = str(uuid.uuid4())
            prediction_log[prediction_id] = {
                "timestamp": time.time(),
                "model_path": loaded_path,
                "latency": latency,
                "n_samples": len(df),
            }
            if monitor is not None:
                monitor.log_data_profile(df, dataset_name=loaded_path or "production")
                if request.ground_truth is not None:
                    y_true = request.ground_truth
                    model_name = loaded_path or "autoforge"
                    monitor.log_prediction(model_name, df, y_true, preds.tolist())
            return {
                "predictions": preds.tolist(),
                "prediction_id": prediction_id,
                "latency_ms": round(latency * 1000, 2),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/feedback")
    def feedback(request: FeedbackRequest):
        if request.prediction_id not in prediction_log:
            raise HTTPException(status_code=404, detail="prediction_id not found")
        entry = prediction_log[request.prediction_id]
        entry["y_true"] = request.y_true
        entry["feedback_received"] = True
        if monitor is not None:
            monitor.accuracy_history.append({
                "timestamp": time.time(),
                "prediction_id": request.prediction_id,
                "y_true": request.y_true,
            })
        return {"status": "recorded", "prediction_id": request.prediction_id}

    @app.get("/monitoring/stats")
    def monitoring_stats():
        if monitor is None:
            return {"monitoring": False, "message": "Monitor not initialized"}
        return monitor.get_monitoring_summary()

    return app


def run_server(model_path: Optional[str] = None, host: str = "0.0.0.0", port: int = 8000):
    if not FASTAPI_AVAILABLE:
        raise ImportError("Install serving extras: pip install 'autoforge[serve]'")
    import uvicorn

    app = create_app(model_path)
    uvicorn.run(app, host=host, port=port)
