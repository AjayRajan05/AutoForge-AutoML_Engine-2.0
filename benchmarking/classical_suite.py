"""
Classical tabular AutoML benchmark suite for AutoForge.

Runs AutoForge (fast/balanced) against sklearn baselines on standard datasets,
validates save/load/predict round-trip, and writes BENCHMARKS.md + JSON results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    fetch_california_housing,
    load_iris,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ClassicalRunResult:
    dataset: str
    task_type: str
    system: str
    cv_or_train_metric: Optional[float]
    holdout_metric: Optional[float]
    metric_name: str
    training_time: float
    preprocessing_recipe: Optional[str]
    winner_model: Optional[str]
    best_params: Optional[Dict[str, Any]]
    report_generated: bool
    predict_roundtrip_ok: bool
    success: bool
    error: Optional[str] = None


def _sklearn_baseline(
    df: pd.DataFrame, target: str, task_type: str, random_state: int = 42
) -> Tuple[float, str, float, str]:
    """Strong sklearn baseline: impute + scale + ordinal + RandomForest."""
    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "enc",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        metric_name = "accuracy"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        metric_name = "r2"
    pipe = Pipeline([("prep", prep), ("model", model)])
    start = time.time()
    pipe.fit(X_train, y_train)
    elapsed = time.time() - start
    preds = pipe.predict(X_test)
    if task_type == "classification":
        holdout = float(accuracy_score(y_test, preds))
    else:
        holdout = float(r2_score(y_test, preds))
    return holdout, metric_name, elapsed, "sklearn_rf_pipeline"


def _predict_roundtrip(automl, df: pd.DataFrame, target: str) -> bool:
    """save_model → load_model → predict on holdout rows."""
    from core.unified_automl import UnifiedAutoML

    holdout = getattr(automl, "_holdout_split_", None) or {}
    X_h = holdout.get("X_holdout")
    if X_h is None or len(X_h) == 0:
        X_h = df.drop(columns=[target]).head(5)
    tmp_name = f"_bench_rt_{int(time.time())}"
    try:
        automl.save_model(tmp_name, overwrite=True)
        loaded = UnifiedAutoML({"verbose": False})
        loaded.load_model(tmp_name)
        preds = loaded.predict(X_h.head(min(5, len(X_h))))
        return preds is not None and len(preds) > 0
    except Exception as exc:
        logger.warning("Round-trip failed: %s", exc)
        return False
    finally:
        model_dir = Path("models") / tmp_name
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)


def _autoforge_run(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    search_depth: str,
    random_state: int = 42,
    test_roundtrip: bool = True,
):
    from core.unified_automl import UnifiedAutoML
    from input_output.input_types import AutoMLInput

    inp = AutoMLInput(
        data=df.copy(),
        target_column=target,
        task_type=task_type,
        search_depth=search_depth,
        random_state=random_state,
        user_preference="fast",
    )
    automl = UnifiedAutoML(
        {
            "search_depth": search_depth,
            "verbose": False,
            "enable_ensemble": search_depth == "balanced",
        }
    )
    start = time.time()
    automl.fit(
        inp,
        enable_optimization=False,
        enable_tracking=False,
        enable_monitoring=False,
    )
    elapsed = time.time() - start
    meta = automl.training_metadata or {}
    holdout = meta.get("holdout_metrics", {})
    if task_type == "classification":
        holdout_metric = holdout.get("accuracy") or holdout.get("primary_score")
        metric_name = "accuracy"
    else:
        holdout_metric = (
            holdout.get("r2_score") or holdout.get("r2") or holdout.get("primary_score")
        )
        metric_name = "r2"
    winner = None
    best_params = {}
    comp = automl.get_model_comparison()
    if comp:
        row = comp[0]
        if not str(row.get("model", "")).startswith("*"):
            winner = str(row.get("model", ""))
            best_params = row.get("best_params") or {}
        elif len(comp) > 1:
            winner = str(comp[1].get("model", ""))
            best_params = comp[1].get("best_params") or {}

    report_ok = False
    try:
        from core.training_report import generate_training_report
        report_ok = len(generate_training_report(automl)) > 100
    except Exception:
        report_ok = False

    rt_ok = _predict_roundtrip(automl, df, target) if test_roundtrip else True

    return (
        float(automl.best_score or 0.0),
        float(holdout_metric) if holdout_metric is not None else None,
        metric_name,
        elapsed,
        meta.get("preprocessing_recipe"),
        winner,
        best_params,
        report_ok,
        rt_ok,
        automl,
    )


def build_standard_datasets() -> Dict[str, Dict[str, Any]]:
    """Return named benchmark datasets as DataFrames."""
    datasets: Dict[str, Dict[str, Any]] = {}

    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=[f"f{i}" for i in range(iris.data.shape[1])])
    iris_df["target"] = iris.target
    datasets["iris"] = {"df": iris_df, "target": "target", "task_type": "classification"}

    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=[f"f{i}" for i in range(wine.data.shape[1])])
    wine_df["target"] = wine.target
    datasets["wine"] = {"df": wine_df, "target": "target", "task_type": "classification"}

    try:
        cal = fetch_california_housing()
        cal_df = pd.DataFrame(cal.data, columns=cal.feature_names)
        cal_df["target"] = cal.target
        datasets["california_housing"] = {
            "df": cal_df,
            "target": "target",
            "task_type": "regression",
        }
    except Exception as exc:
        logger.warning("California housing unavailable: %s", exc)

    X_reg, y_reg = make_regression(
        n_samples=400, n_features=12, n_informative=8, noise=1.0, random_state=42
    )
    reg_df = pd.DataFrame(X_reg, columns=[f"f{i}" for i in range(X_reg.shape[1])])
    reg_df["target"] = y_reg
    datasets["regression_synthetic"] = {
        "df": reg_df,
        "target": "target",
        "task_type": "regression",
    }

    X_clf, y_clf = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        weights=[0.85, 0.10, 0.05],
        random_state=42,
    )
    clf_df = pd.DataFrame(X_clf, columns=[f"f{i}" for i in range(X_clf.shape[1])])
    clf_df["target"] = y_clf
    datasets["imbalanced_classification"] = {
        "df": clf_df,
        "target": "target",
        "task_type": "classification",
    }

    rng = np.random.RandomState(42)
    n = 500
    high_card_df = pd.DataFrame(
        {
            "num1": rng.randn(n),
            "num2": rng.randn(n),
            "city": [f"city_{rng.randint(0, 80)}" for _ in range(n)],
            "segment": [f"seg_{rng.randint(0, 5)}" for _ in range(n)],
        }
    )
    high_card_df["target"] = (
        (high_card_df["num1"] + high_card_df["num2"] * 0.5 + rng.randn(n) * 0.2) > 0
    ).astype(int)
    datasets["high_cardinality_cats"] = {
        "df": high_card_df,
        "target": "target",
        "task_type": "classification",
    }

    housing_path = Path("data.csv")
    if housing_path.exists():
        try:
            hdf = pd.read_csv(housing_path)
            if "price" in hdf.columns:
                datasets["housing_csv"] = {
                    "df": hdf,
                    "target": "price",
                    "task_type": "regression",
                }
        except Exception as exc:
            logger.warning("Could not load data.csv: %s", exc)

    return datasets


def run_classical_suite(
    datasets: Optional[List[str]] = None,
    depths: Optional[List[str]] = None,
    include_baseline: bool = True,
    test_roundtrip: bool = True,
) -> List[ClassicalRunResult]:
    """Run classical benchmark suite and return structured results."""
    all_datasets = build_standard_datasets()
    dataset_names = datasets or list(all_datasets.keys())
    depths = depths or ["fast", "balanced"]
    results: List[ClassicalRunResult] = []

    for name in dataset_names:
        if name not in all_datasets:
            logger.warning("Unknown dataset %s, skipping", name)
            continue
        spec = all_datasets[name]
        df = spec["df"]
        target = spec["target"]
        task_type = spec["task_type"]

        if include_baseline:
            try:
                holdout, metric_name, elapsed, recipe = _sklearn_baseline(
                    df, target, task_type
                )
                results.append(
                    ClassicalRunResult(
                        dataset=name,
                        task_type=task_type,
                        system="sklearn_baseline",
                        cv_or_train_metric=None,
                        holdout_metric=holdout,
                        metric_name=metric_name,
                        training_time=elapsed,
                        preprocessing_recipe=recipe,
                        winner_model="RandomForest",
                        best_params={},
                        report_generated=False,
                        predict_roundtrip_ok=True,
                        success=True,
                    )
                )
            except Exception as exc:
                results.append(
                    ClassicalRunResult(
                        dataset=name,
                        task_type=task_type,
                        system="sklearn_baseline",
                        cv_or_train_metric=None,
                        holdout_metric=None,
                        metric_name="accuracy" if task_type == "classification" else "r2",
                        training_time=0.0,
                        preprocessing_recipe=None,
                        winner_model=None,
                        best_params={},
                        report_generated=False,
                        predict_roundtrip_ok=False,
                        success=False,
                        error=str(exc),
                    )
                )

        for depth in depths:
            try:
                (
                    cv_score,
                    holdout,
                    metric_name,
                    elapsed,
                    recipe,
                    winner,
                    best_params,
                    report_ok,
                    rt_ok,
                    _,
                ) = _autoforge_run(
                    df, target, task_type, depth, test_roundtrip=test_roundtrip
                )
                results.append(
                    ClassicalRunResult(
                        dataset=name,
                        task_type=task_type,
                        system=f"autoforge_{depth}",
                        cv_or_train_metric=cv_score,
                        holdout_metric=holdout,
                        metric_name=metric_name,
                        training_time=elapsed,
                        preprocessing_recipe=recipe,
                        winner_model=winner,
                        best_params=best_params,
                        report_generated=report_ok,
                        predict_roundtrip_ok=rt_ok,
                        success=True,
                    )
                )
            except Exception as exc:
                logger.exception("AutoForge %s failed on %s", depth, name)
                results.append(
                    ClassicalRunResult(
                        dataset=name,
                        task_type=task_type,
                        system=f"autoforge_{depth}",
                        cv_or_train_metric=None,
                        holdout_metric=None,
                        metric_name="accuracy" if task_type == "classification" else "r2",
                        training_time=0.0,
                        preprocessing_recipe=None,
                        winner_model=None,
                        best_params={},
                        report_generated=False,
                        predict_roundtrip_ok=False,
                        success=False,
                        error=str(exc),
                    )
                )
    return results


def evaluate_exit_criteria(results: List[ClassicalRunResult]) -> Dict[str, Any]:
    """Check benchmark exit criteria from the perfection roadmap."""
    by_dataset: Dict[str, Dict[str, ClassicalRunResult]] = {}
    for r in results:
        by_dataset.setdefault(r.dataset, {})[r.system] = r

    balanced_beats_fast = 0
    compared = 0
    baseline_close = 0
    baseline_total = 0
    roundtrip_failures = []

    for dataset, systems in by_dataset.items():
        fast = systems.get("autoforge_fast")
        balanced = systems.get("autoforge_balanced")
        baseline = systems.get("sklearn_baseline")
        if fast and balanced and fast.success and balanced.success:
            compared += 1
            if (balanced.holdout_metric or 0) >= (fast.holdout_metric or 0):
                balanced_beats_fast += 1
        if balanced and baseline and balanced.success and baseline.success:
            baseline_total += 1
            b = baseline.holdout_metric or 0
            a = balanced.holdout_metric or 0
            if b == 0:
                if a >= b:
                    baseline_close += 1
            elif abs(a - b) / abs(b) <= 0.05 or a >= b:
                baseline_close += 1
        for sys_name, row in systems.items():
            if sys_name.startswith("autoforge") and not row.predict_roundtrip_ok:
                roundtrip_failures.append(f"{dataset}/{sys_name}")

    return {
        "balanced_beats_fast": f"{balanced_beats_fast}/{compared}",
        "balanced_within_5pct_baseline": f"{baseline_close}/{baseline_total}",
        "predict_roundtrip_failures": roundtrip_failures,
        "all_roundtrip_ok": len(roundtrip_failures) == 0,
    }


def write_benchmarks_md(
    results: List[ClassicalRunResult],
    output_dir: str = "benchmarks/results",
) -> Path:
    """Write BENCHMARKS.md and results.json under output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    md_path = out / "BENCHMARKS.md"

    payload = [asdict(r) for r in results]
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    criteria = evaluate_exit_criteria(results)
    lines = [
        "# AutoForge Classical Benchmarks",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Exit criteria",
        "",
        f"- Balanced beats fast (holdout): **{criteria['balanced_beats_fast']}**",
        f"- Balanced within 5% of sklearn baseline: **{criteria['balanced_within_5pct_baseline']}**",
        f"- Predict round-trip OK: **{criteria['all_roundtrip_ok']}**",
        "",
        "## Summary",
        "",
        "| Dataset | Task | System | Holdout | CV | Time (s) | Recipe | Report | RT | Winner |",
        "|---------|------|--------|---------|-----|----------|--------|--------|-----|--------|",
    ]
    for r in results:
        holdout = (
            f"{r.holdout_metric:.4f}"
            if r.holdout_metric is not None
            else ("FAIL" if not r.success else "N/A")
        )
        cv = f"{r.cv_or_train_metric:.4f}" if r.cv_or_train_metric is not None else "—"
        recipe = r.preprocessing_recipe or "—"
        winner = (r.winner_model or "—")[:32]
        report = "yes" if r.report_generated else "no"
        rt = "ok" if r.predict_roundtrip_ok else "FAIL"
        status = "" if r.success else f" ({r.error})"
        lines.append(
            f"| {r.dataset} | {r.task_type} | {r.system} | {holdout} | {cv} | "
            f"{r.training_time:.2f} | {recipe} | {report} | {rt} | {winner}{status} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Holdout: AutoForge internal 20% split (feature selection on train only).",
            "- Baseline: sklearn Pipeline (impute + scale/ordinal + RandomForest).",
            "- RT: save_model → load_model → predict round-trip.",
            "- Re-run: `python -m benchmarking.classical_suite`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s and %s", md_path, json_path)
    return md_path


def main():
    logging.basicConfig(level=logging.INFO)
    results = run_classical_suite(depths=["fast", "balanced"])
    path = write_benchmarks_md(results)
    criteria = evaluate_exit_criteria(results)
    print(f"Benchmark report written to {path}")
    print(f"Exit criteria: {criteria}")


if __name__ == "__main__":
    main()
