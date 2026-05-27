"""
Unified preprocessing pipeline — single authority for train/predict transforms.
Uses sklearn ColumnTransformer when available; legacy artifact replay supported.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from utils.dtype_helpers import categorical_columns, numeric_columns

logger = logging.getLogger(__name__)


class _TargetMeanEncoder:
    """Mean target encoding for low-cardinality categoricals (fit on train only)."""

    def __init__(self):
        self.maps_: Dict[str, Dict[Any, float]] = {}
        self.global_mean_: float = 0.0

    def fit(self, X, y=None):
        import pandas as pd

        self.global_mean_ = float(np.mean(y)) if y is not None else 0.0
        frame = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y_s = pd.Series(y).reset_index(drop=True)
        for col in frame.columns:
            stats = y_s.groupby(frame[col].astype(str)).mean()
            self.maps_[col] = stats.to_dict()
        return self

    def transform(self, X):
        import pandas as pd

        frame = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        out = frame.copy()
        for col in frame.columns:
            mapping = self.maps_.get(col, {})
            out[col] = frame[col].astype(str).map(mapping).fillna(self.global_mean_)
        return out.to_numpy()


class PreprocessingPipeline:
    """Fit once on training data; transform at predict time via saved artifacts."""

    def __init__(
        self,
        scale_features: bool = True,
        scale_type: str = "standard",
        feature_engineering: bool = False,
        polynomial_degree: int = 2,
        impute_strategy: str = "median",
        encoding_strategy: str = "ordinal",
        recipe_name: Optional[str] = None,
        max_categories: int = 10,
    ):
        self.scale_features = scale_features
        self.scale_type = scale_type
        self.feature_engineering = feature_engineering
        self.polynomial_degree = polynomial_degree
        self.impute_strategy = impute_strategy
        self.encoding_strategy = encoding_strategy
        self.recipe_name = recipe_name
        self.max_categories = max_categories
        self.artifacts: Dict[str, Any] = {}
        self.is_fitted = False
        self.column_transformer: Optional[ColumnTransformer] = None

    def _numeric_imputer(self) -> SimpleImputer:
        strategy = self.impute_strategy if self.impute_strategy in ("mean", "median") else "median"
        return SimpleImputer(strategy=strategy)

    def _categorical_imputer(self) -> SimpleImputer:
        return SimpleImputer(strategy="most_frequent")

    def _scaler(self):
        if not self.scale_features or self.scale_type == "none":
            return None
        if self.scale_type == "robust":
            return RobustScaler()
        if self.scale_type == "quantile":
            return QuantileTransformer(output_distribution="normal", random_state=42)
        return StandardScaler()

    def _detect_columns(self, X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        X_work = X.copy()
        datetime_cols = X_work.select_dtypes(include=["datetime64"]).columns.tolist()
        for col in datetime_cols:
            X_work[col] = pd.to_datetime(X_work[col]).astype("int64") / 10**9
        numeric_cols = numeric_columns(X_work)
        categorical_cols = categorical_columns(X_work)
        return numeric_cols, categorical_cols, datetime_cols

    def _build_column_transformer(
        self, numeric_cols: List[str], categorical_cols: List[str]
    ) -> ColumnTransformer:
        transformers = []
        if numeric_cols:
            num_steps: List[Tuple[str, Any]] = [
                ("imputer", self._numeric_imputer()),
            ]
            if self.feature_engineering:
                num_steps.append(
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=self.polynomial_degree,
                            include_bias=False,
                            interaction_only=True,
                        ),
                    )
                )
            scaler = self._scaler()
            if scaler is not None:
                num_steps.append(("scaler", scaler))
            transformers.append(("num", Pipeline(num_steps), numeric_cols))

        if categorical_cols:
            cat_steps: List[Tuple[str, Any]] = [
                ("imputer", self._categorical_imputer()),
            ]
            if self.encoding_strategy == "ordinal":
                cat_steps.append(
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    )
                )
            elif self.encoding_strategy == "onehot":
                cat_steps.append(
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            max_categories=self.max_categories,
                            sparse_output=False,
                        ),
                    )
                )
            elif self.encoding_strategy != "target":
                cat_steps.append(
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    )
                )
            transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

        if not transformers:
            transformers.append(
                (
                    "passthrough",
                    "passthrough",
                    list(numeric_cols + categorical_cols) or list(range(0)),
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _frame_from_transform(
        self, X_arr: np.ndarray, feature_names: List[str]
    ) -> pd.DataFrame:
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        cols = feature_names[: X_arr.shape[1]]
        if len(cols) < X_arr.shape[1]:
            cols = cols + [f"feature_{i}" for i in range(len(cols), X_arr.shape[1])]
        return pd.DataFrame(X_arr, columns=cols[: X_arr.shape[1]], index=None)

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, task_type: str = "classification"
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        X_work = X.copy()
        y_processed = y.copy()
        numeric_cols, categorical_cols, datetime_cols = self._detect_columns(X_work)
        for col in datetime_cols:
            X_work[col] = pd.to_datetime(X_work[col]).astype("int64") / 10**9

        self.column_transformer = self._build_column_transformer(
            numeric_cols, categorical_cols
        )
        if not numeric_cols and not categorical_cols:
            X_processed = X_work.copy()
            feature_names = X_processed.columns.tolist()
        else:
            X_arr = self.column_transformer.fit_transform(X_work)
            try:
                feature_names = list(
                    self.column_transformer.get_feature_names_out()
                )
            except Exception:
                feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
            X_processed = self._frame_from_transform(X_arr, feature_names)

        if self.encoding_strategy == "target" and categorical_cols:
            self._target_encoder = _TargetMeanEncoder()
            cat_frame = X_work[categorical_cols]
            self._target_encoder.fit(cat_frame, y_processed)
            encoded = self._target_encoder.transform(cat_frame)
            enc_df = pd.DataFrame(encoded, columns=categorical_cols, index=X_processed.index)
            for col in categorical_cols:
                if col in X_processed.columns:
                    X_processed[col] = enc_df[col].values
                else:
                    X_processed = pd.concat([X_processed, enc_df[[col]]], axis=1)

        info: Dict[str, Any] = {
            "scale_features": self.scale_features,
            "scale_type": self.scale_type,
            "impute_strategy": self.impute_strategy,
            "encoding_strategy": self.encoding_strategy,
            "recipe_name": self.recipe_name,
            "feature_engineering": self.feature_engineering,
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "datetime_features": datetime_cols,
            "feature_columns": list(X.columns),
            "output_feature_names": X_processed.columns.tolist(),
            "column_transformer": self.column_transformer,
            "original_shape": X.shape,
            "processed_shape": X_processed.shape,
            "task_type": task_type,
            "pipeline_type": "column_transformer",
        }
        if self.encoding_strategy == "target" and categorical_cols:
            info["target_encoder_maps"] = self._target_encoder.maps_
            info["target_encoder_global_mean"] = self._target_encoder.global_mean_

        if task_type == "classification" and y_processed.dtype == "object":
            le_target = LabelEncoder()
            y_processed = pd.Series(le_target.fit_transform(y_processed))
            info["label_encoder"] = le_target

        self.artifacts = info
        self.is_fitted = True
        return X_processed, y_processed, info

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted and not self.artifacts:
            raise ValueError("PreprocessingPipeline must be fitted before transform")

        steps = self.artifacts
        if steps.get("pipeline_type") == "column_transformer" and steps.get(
            "column_transformer"
        ):
            ct = steps["column_transformer"]
            X_work = X.copy()
            for col in steps.get("datetime_features", []):
                if col in X_work.columns:
                    X_work[col] = pd.to_datetime(X_work[col]).astype("int64") / 10**9
            X_arr = ct.transform(X_work)
            names = steps.get("output_feature_names") or []
            X_out = self._frame_from_transform(X_arr, names)
            if steps.get("encoding_strategy") == "target" and steps.get("categorical_features"):
                encoder = getattr(self, "_target_encoder", None)
                if encoder is None and "target_encoder_maps" in steps:
                    encoder = _TargetMeanEncoder()
                    encoder.maps_ = steps["target_encoder_maps"]
                    encoder.global_mean_ = steps.get("target_encoder_global_mean", 0.0)
                if encoder is not None:
                    cat_cols = steps["categorical_features"]
                    cat_frame = X_work[[c for c in cat_cols if c in X_work.columns]]
                    if not cat_frame.empty:
                        encoded = encoder.transform(cat_frame)
                        enc_df = pd.DataFrame(encoded, columns=cat_frame.columns, index=X_out.index)
                        for col in cat_frame.columns:
                            if col in X_out.columns:
                                X_out[col] = enc_df[col].values
                            else:
                                X_out = pd.concat([X_out, enc_df[[col]]], axis=1)
            return X_out

        return self._legacy_transform(X, steps)

    def _legacy_transform(self, X: pd.DataFrame, steps: Dict[str, Any]) -> pd.DataFrame:
        X_processed = X.copy()
        if steps.get("missing_values_handled"):
            for col in steps.get("numeric_features", []):
                if col in X_processed.columns:
                    X_processed[col] = X_processed[col].fillna(
                        X_processed[col].mean()
                    )
            for col in steps.get("categorical_features", []):
                if col in X_processed.columns:
                    X_processed[col] = X_processed[col].fillna("unknown")

        for col in steps.get("categorical_features", []):
            if col not in X_processed.columns:
                continue
            encoders = steps.get("categorical_encoding", {})
            if col in encoders and hasattr(encoders[col], "transform"):
                values = X_processed[col].astype(str)
                known = set(encoders[col].classes_)
                fallback = encoders[col].classes_[0]
                values = values.where(values.isin(known), fallback)
                X_processed[col] = encoders[col].transform(values)

        numeric_features = steps.get("numeric_features", [])
        scaler = steps.get("scaler")
        if scaler is not None and numeric_features:
            cols = [c for c in numeric_features if c in X_processed.columns]
            if cols:
                X_processed[cols] = scaler.transform(X_processed[cols])

        expected = steps.get("feature_columns", [])
        if expected:
            for col in expected:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            X_processed = X_processed[expected]
        return X_processed

    def save_artifacts(self, path: str) -> str:
        joblib.dump(self.artifacts, path)
        return path

    @classmethod
    def load_artifacts(cls, path: str) -> "PreprocessingPipeline":
        pipe = cls()
        pipe.artifacts = joblib.load(path)
        pipe.is_fitted = True
        pipe.scale_features = pipe.artifacts.get("scale_features", True)
        pipe.scale_type = pipe.artifacts.get("scale_type", "standard")
        pipe.impute_strategy = pipe.artifacts.get("impute_strategy", "median")
        pipe.encoding_strategy = pipe.artifacts.get("encoding_strategy", "ordinal")
        pipe.recipe_name = pipe.artifacts.get("recipe_name")
        pipe.feature_engineering = pipe.artifacts.get("feature_engineering", False)
        pipe.column_transformer = pipe.artifacts.get("column_transformer")
        return pipe

    @classmethod
    def from_recipe(cls, recipe_name: str, recipe_config: Optional[Dict[str, Any]] = None) -> "PreprocessingPipeline":
        """Build a pipeline instance from a named preprocessing recipe."""
        cfg = dict(recipe_config or {})
        cfg["recipe_name"] = recipe_name
        return cls(**{k: v for k, v in cfg.items() if k in (
            "scale_features", "scale_type", "feature_engineering", "polynomial_degree",
            "impute_strategy", "encoding_strategy", "recipe_name", "max_categories",
        )})
