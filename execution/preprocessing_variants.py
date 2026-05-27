"""
Evaluate models on raw vs fully-processed data variants (legacy engine pattern).
"""

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd
from sklearn.model_selection import cross_val_score

from execution.preprocessing_pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


def evaluate_preprocessing_variants(
    model_factory: Callable,
    X_raw: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    cv_folds: int = 3,
    n_jobs: int = -1,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Score default model on raw features vs pipeline-processed features.
    Returns best variant metadata and score.
    """
    if task_type == "classification":
        scoring = "accuracy"
    else:
        scoring = "neg_mean_squared_error"

    variants = {"raw": X_raw.copy()}
    try:
        pipe = PreprocessingPipeline(scale_features=scale_features)
        X_proc, _, _ = pipe.fit_transform(X_raw, y, task_type=task_type)
        variants["processed"] = X_proc
    except Exception as exc:
        logger.warning("Processed variant skipped: %s", exc)

    best_score = float("-inf")
    best_variant = "raw"
    scores: Dict[str, float] = {}

    for name, X_variant in variants.items():
        try:
            model = model_factory()
            cv_scores = cross_val_score(
                model, X_variant, y, cv=cv_folds, scoring=scoring, n_jobs=n_jobs
            )
            mean_score = float(cv_scores.mean())
            scores[name] = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_variant = name
        except Exception as exc:
            logger.debug("Variant %s failed: %s", name, exc)

    return {
        "best_variant": best_variant,
        "best_score": best_score,
        "variant_scores": scores,
    }
