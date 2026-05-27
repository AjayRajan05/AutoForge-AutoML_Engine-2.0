"""End-to-end: train → save → TestClient predict."""

import os
import sys

import pandas as pd
import pytest
from sklearn.datasets import make_classification

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput
from serving.app import create_app


@pytest.fixture
def clf_df():
    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        random_state=7,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def test_train_save_serve_predict(clf_df, tmp_path):
    inp = AutoMLInput(
        data=clf_df,
        target_column="target",
        task_type="classification",
        search_depth="fast",
        random_state=7,
    )
    automl = UnifiedAutoML({"search_depth": "fast", "verbose": False, "enable_ensemble": False})
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)

    model_dir = automl.save_model(str(tmp_path / "serve_model"))
    app = create_app(model_path=model_dir)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    row = clf_df.drop(columns=["target"]).iloc[0].to_dict()
    resp = client.post("/predict", json={"records": [row]})
    assert resp.status_code == 200
    body = resp.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1
