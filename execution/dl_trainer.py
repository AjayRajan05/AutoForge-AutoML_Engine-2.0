"""
Deep learning trainer using Keras Tuner (optional dependency).
"""

import logging
import os
import random
import shutil
import string
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DL_AVAILABLE = False
try:
    import tensorflow as tf
    from kerastuner import HyperModel
    from kerastuner.tuners import RandomSearch
    from tensorflow import keras
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
    from tensorflow.keras.models import Sequential

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    DL_AVAILABLE = True
except ImportError:
    tf = None
    HyperModel = object
    RandomSearch = None
    keras = None


def require_dl_dependencies():
    if not DL_AVAILABLE:
        raise ImportError(
            "Deep learning requires optional dependencies. Install with: pip install 'autoforge[dl]'"
        )


def _load_hyper_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "config" / "search_hypers.yml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


class KerasModelWrapper:
    """Sklearn-compatible wrapper around a Keras model."""

    def __init__(self, model, task_type: str = "classification", name: str = "dnn"):
        self.model = model
        self.task_type = task_type
        self.name = name
        self.backend = "keras"

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        preds = self.model.predict(X, verbose=0)
        if self.task_type == "classification":
            if preds.ndim > 1 and preds.shape[1] > 1:
                return np.argmax(preds, axis=1)
            return (preds.ravel() >= 0.5).astype(int)
        return preds.ravel()

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        preds = self.model.predict(X, verbose=0)
        if preds.ndim == 1:
            preds = np.column_stack([1 - preds, preds])
        return preds


if DL_AVAILABLE:

    class DNNSearch(HyperModel):
        def __init__(self, n_classes: int, hyper_config: Dict[str, Any], use_dropout: bool = False):
            self.n_classes = n_classes
            self.hyper_config = hyper_config
            self.use_dropout = use_dropout
            self.name = "DNN"

        def build(self, hp):
            dnn_cfg = self.hyper_config.get("DNN", {})
            n_layers = hp.Int(
                "n_layers",
                dnn_cfg.get("n_layers", {}).get("min_value", 1),
                dnn_cfg.get("n_layers", {}).get("max_value", 2),
                step=dnn_cfg.get("n_layers", {}).get("step", 1),
            )
            model = Sequential()
            for i in range(n_layers):
                units = hp.Int(
                    f"n_units_{i}",
                    dnn_cfg.get("n_units", {}).get("min_value", 32),
                    dnn_cfg.get("n_units", {}).get("max_value", 512),
                    step=dnn_cfg.get("n_units", {}).get("step", 32),
                )
                model.add(Dense(units, activation="relu"))
                if self.use_dropout:
                    model.add(Dropout(0.3))
            if self.n_classes >= 2:
                model.add(Dense(self.n_classes, activation="softmax"))
            else:
                model.add(Dense(1))
            if self.n_classes == 1:
                model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mse"])
            else:
                model.compile(
                    optimizer=keras.optimizers.Adam(0.001),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
            return model

    class LSTMSearch(HyperModel):
        """LSTM architecture for sequential/tabular data reshaped as sequences."""

        def __init__(self, n_classes: int, hyper_config: Dict[str, Any]):
            self.n_classes = n_classes
            self.hyper_config = hyper_config
            self.name = "LSTM"

        def build(self, hp):
            lstm_cfg = self.hyper_config.get("LSTM", {})
            units = hp.Int(
                "lstm_units",
                lstm_cfg.get("n_units", {}).get("min_value", 32),
                lstm_cfg.get("n_units", {}).get("max_value", 128),
                step=lstm_cfg.get("n_units", {}).get("step", 32),
            )
            model = Sequential()
            model.add(LSTM(units, input_shape=(1, None)))
            if self.n_classes >= 2:
                model.add(Dense(self.n_classes, activation="softmax"))
            else:
                model.add(Dense(1))
            if self.n_classes == 1:
                model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mse"])
            else:
                model.compile(
                    optimizer=keras.optimizers.Adam(0.001),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
            return model

    class CNNSearch(HyperModel):
        """1D CNN for tabular features treated as a sequence."""

        def __init__(self, n_classes: int, hyper_config: Dict[str, Any]):
            self.n_classes = n_classes
            self.hyper_config = hyper_config
            self.name = "CNN"

        def build(self, hp):
            cnn_cfg = self.hyper_config.get("CNN", {})
            filters = hp.Int(
                "filters",
                cnn_cfg.get("filters", {}).get("min_value", 16),
                cnn_cfg.get("filters", {}).get("max_value", 64),
                step=cnn_cfg.get("filters", {}).get("step", 16),
            )
            model = Sequential()
            model.add(Conv1D(filters, kernel_size=3, activation="relu", input_shape=(None, 1)))
            model.add(Flatten())
            if self.n_classes >= 2:
                model.add(Dense(self.n_classes, activation="softmax"))
            else:
                model.add(Dense(1))
            if self.n_classes == 1:
                model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mse"])
            else:
                model.compile(
                    optimizer=keras.optimizers.Adam(0.001),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
            return model


class DLTrainer:
    """Train Keras models via RandomSearch and return AutoForge model_results entries."""

    def __init__(self, models_path: Optional[str] = None):
        self.models_path = models_path or tempfile.mkdtemp(prefix="autoforge_dl_")
        self.hyper_config = _load_hyper_config()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        max_trials: int = 5,
        epochs: int = 10,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        require_dl_dependencies()
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)
        if task_type == "regression":
            n_classes = 1
            objective = "val_loss"
        else:
            n_classes = len(np.unique(y_arr))
            objective = "val_accuracy"

        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=random_state,
            stratify=y_arr if task_type == "classification" and n_classes > 1 else None,
        )

        search_dir = os.path.join(self.models_path, "tuner")
        os.makedirs(search_dir, exist_ok=True)

        architectures = [
            ("dnn", DNNSearch),
            ("lstm", LSTMSearch),
            ("cnn", CNNSearch),
        ]

        model_results: Dict[str, Any] = {}
        for arch_name, arch_cls in architectures:
            try:
                if arch_name == "lstm":
                    X_tr = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                    X_va = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
                elif arch_name == "cnn":
                    X_tr = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    X_va = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                else:
                    X_tr, X_va = X_train, X_val

                project_name = f"{arch_name}_{''.join(random.choices(string.ascii_letters, k=8))}"
                hyper_model = arch_cls(n_classes, self.hyper_config)
                tuner = RandomSearch(
                    hyper_model,
                    objective=objective,
                    max_trials=max_trials,
                    executions_per_trial=1,
                    directory=search_dir,
                    project_name=project_name,
                )
                tuner.search(
                    X_tr, y_train, epochs=epochs,
                    validation_data=(X_va, y_val), verbose=0,
                )
                best_models = tuner.get_best_models(num_models=1)
                if not best_models:
                    continue
                keras_model = best_models[0]
                wrapper = KerasModelWrapper(
                    keras_model, task_type=task_type, name=arch_name
                )
                y_pred = wrapper.predict(X_va)
                if task_type == "classification":
                    score = accuracy_score(y_val, y_pred)
                    scoring = "accuracy"
                else:
                    score = r2_score(y_val, y_pred)
                    scoring = "r2_score"
                model_name = f"{arch_name}_{score:.4f}"
                model_results[model_name] = {
                    "model": wrapper,
                    "cv_scores": [score],
                    "cv_mean": score,
                    "cv_std": 0.0,
                    "scoring": scoring,
                    "backend": "keras",
                    "architecture": arch_name,
                }
                logger.info("DL %s: %.4f", model_name, score)
                shutil.rmtree(os.path.join(search_dir, project_name), ignore_errors=True)
            except Exception as exc:
                logger.warning("DL architecture %s skipped: %s", arch_name, exc)

        return model_results
