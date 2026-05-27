"""Training report bundle and holdout metrics."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput


@pytest.fixture
def clf_df():
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture
def reg_df():
    X, y = make_regression(n_samples=60, n_features=5, noise=0.5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def _fit_fast(df, task_type):
    inp = AutoMLInput(
        data=df,
        target_column="target",
        task_type=task_type,
        search_depth="fast",
        random_state=42,
    )
    automl = UnifiedAutoML({"search_depth": "fast", "verbose": False, "enable_ensemble": False})
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    return automl


def test_save_produces_report_and_leaderboard(clf_df, tmp_path):
    automl = _fit_fast(clf_df, "classification")
    automl.config["model_save_path"] = str(tmp_path)
    automl.model_saver.base_path = tmp_path
    model_dir = automl.save_model("report_test")
    base = Path(model_dir)
    assert (base / "REPORT.md").exists()
    assert (base / "leaderboard.csv").exists()
    assert (base / "selection_decision.json").exists()
    assert (base / "preprocessing_report.json").exists()


def test_get_selection_report_has_winner_and_leaderboard(clf_df):
    automl = _fit_fast(clf_df, "classification")
    report = automl.get_selection_report()
    assert "selection" in report
    assert report["selection"].get("winner")
    assert len(report.get("leaderboard", [])) >= 1
    assert "preprocessing" in report


def test_selection_report_includes_preprocessing_recipe():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=70, n_features=6, random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    inp = AutoMLInput(
        data=df,
        target_column="target",
        task_type="classification",
        search_depth="balanced",
        max_trials=2,
        random_state=1,
    )
    automl = UnifiedAutoML(
        {
            "search_depth": "balanced",
            "verbose": False,
            "enable_ensemble": False,
            "use_processors": False,
        }
    )
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    report = automl.get_selection_report()
    assert report["preprocessing"].get("recipe") is not None
    assert report["preprocessing"].get("search_results")


def test_selection_decision_ensemble_fields(clf_df):
    automl = _fit_fast(clf_df, "classification")
    decision = automl.training_metadata.get("selection_decision") or {}
    if not decision:
        from core.training_report import build_selection_decision
        decision = build_selection_decision(automl)
    assert "ensemble_tried" in decision
    assert "ensemble_accepted" in decision
    assert "score_margin" in decision
    assert "reason" in decision


def test_ensemble_attempted_on_close_leaderboard(clf_df, monkeypatch):
    """When top models are close, balanced search attempts an ensemble."""
    inp = AutoMLInput(
        data=clf_df,
        target_column="target",
        task_type="classification",
        search_depth="balanced",
        max_trials=1,
        random_state=42,
    )
    automl = UnifiedAutoML(
        {
            "search_depth": "balanced",
            "verbose": False,
            "enable_ensemble": True,
            "use_processors": False,
            "ensemble_epsilon": 1.0,
        }
    )

    class FakeEnsembleIntegrator:
        def create_voting_ensemble(self, models, task_type, voting="hard"):
            from sklearn.ensemble import VotingClassifier
            names = [f"m{i}" for i in range(len(models))]
            return VotingClassifier(list(zip(names, models)), voting="hard")

        def evaluate_ensemble(self, ensemble, X, y, individual_models=None):
            return {
                "ensemble_score": 0.99,
                "tried": True,
                "ensemble_metrics": {"accuracy": 0.99},
            }

    automl.ensemble_integrator = FakeEnsembleIntegrator()
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    info = automl.training_metadata.get("ensemble_info") or {}
    assert info.get("tried") is True


def test_holdout_metrics_in_training_metadata(clf_df, reg_df):
    automl_clf = _fit_fast(clf_df, "classification")
    holdout_clf = automl_clf.training_metadata.get("holdout_metrics", {})
    assert holdout_clf
    assert "accuracy" in holdout_clf or "primary_score" in holdout_clf

    automl_reg = _fit_fast(reg_df, "regression")
    holdout_reg = automl_reg.training_metadata.get("holdout_metrics", {})
    reg_metrics = automl_reg.training_metadata.get("regression_metrics", {})
    assert holdout_reg
    assert "r2_score" in holdout_reg or "r2_score" in holdout_reg
    assert reg_metrics.get("cv_neg_mse") is not None
    assert "r2_score" in reg_metrics or "rmse" in reg_metrics
