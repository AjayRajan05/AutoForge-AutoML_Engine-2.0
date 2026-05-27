"""
Training justification reports — selection summary, markdown, and save bundle.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.unified_automl import UnifiedAutoML


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def build_selection_decision(automl: "UnifiedAutoML") -> Dict[str, Any]:
    """Build structured winner/runner-up/ensemble decision payload."""
    meta = automl.training_metadata or {}
    rows = automl.get_model_comparison()
    leaderboard = [
        r for r in rows
        if not str(r.get("model", "")).startswith("*")
    ]
    winner_row = leaderboard[0] if leaderboard else {}
    runner_up = leaderboard[1] if len(leaderboard) > 1 else None
    winner_score = float(winner_row.get("cv_mean", automl.best_score or 0.0))
    runner_score = float(runner_up.get("cv_mean", 0.0)) if runner_up else None
    margin = (winner_score - runner_score) if runner_score is not None else None

    ensemble_info = meta.get("ensemble_info") or {}
    ensemble_method = meta.get("ensemble_method")
    ensemble_tried = bool(ensemble_method or ensemble_info.get("tried"))
    if not ensemble_tried and meta.get("ensemble_run_dir"):
        ensemble_tried = True
    ensemble_selected = "ensemble" in str(type(automl.best_model).__name__).lower() or bool(
        meta.get("ensemble_run_dir") and meta.get("ensemble_method")
    )
    if ensemble_info.get("selected") is True:
        ensemble_selected = True
    if ensemble_info.get("tried") is False:
        ensemble_tried = False

    if ensemble_selected:
        ensemble_reason = f"Ensemble ({ensemble_method or 'unknown'}) beat single-model CV score."
    elif ensemble_tried and not ensemble_selected:
        ensemble_reason = ensemble_info.get(
            "skip_reason",
            "Ensemble did not improve over best single model.",
        )
    elif not ensemble_tried:
        ensemble_reason = ensemble_info.get(
            "skip_reason",
            "Ensemble not attempted (models not close enough or disabled).",
        )
    else:
        ensemble_reason = "Single model retained."

    return {
        "winner": {
            "model": winner_row.get("model", type(automl.best_model).__name__),
            "cv_mean": winner_score,
            "cv_std": winner_row.get("cv_std"),
            "best_params": winner_row.get("best_params", {}),
            "backend": winner_row.get("backend", "sklearn"),
        },
        "runner_up": (
            {
                "model": runner_up.get("model"),
                "cv_mean": runner_score,
                "cv_std": runner_up.get("cv_std"),
            }
            if runner_up
            else None
        ),
        "margin": margin,
        "score_margin": margin,
        "ensemble_tried": ensemble_tried,
        "ensemble_accepted": ensemble_selected,
        "reason": ensemble_reason,
        "selected_model_type": type(automl.best_model).__name__ if automl.best_model else None,
        "ensemble": {
            "tried": ensemble_tried,
            "selected": ensemble_selected,
            "accepted": ensemble_selected,
            "method": ensemble_method,
            "reason": ensemble_reason,
        },
        "task_type": meta.get("task_type"),
        "search_depth": getattr(automl, "search_depth_", None),
        "timestamp": datetime.now().isoformat(),
    }


def build_preprocessing_report(automl: "UnifiedAutoML") -> Dict[str, Any]:
    """Serializable preprocessing summary for JSON export."""
    meta = automl.training_metadata or {}
    artifacts = meta.get("preprocessing_artifacts") or meta.get("preprocessing_steps") or {}
    safe = {}
    for key, value in artifacts.items():
        if key == "column_transformer":
            safe["column_transformer"] = "ColumnTransformer(fitted)"
            safe["pipeline_type"] = artifacts.get("pipeline_type", "column_transformer")
        elif hasattr(value, "__class__") and not isinstance(value, (dict, list, str, int, float, bool)):
            safe[key] = type(value).__name__
        else:
            safe[key] = value
    profile = None
    if automl.current_profile and hasattr(automl.current_profile, "__dict__"):
        profile = {
            k: _json_safe(v)
            for k, v in automl.current_profile.__dict__.items()
            if k not in ("recommendations",)
        }
    return {
        "data_type": meta.get("data_type"),
        "task_type": meta.get("task_type"),
        "preprocessing_recipe": meta.get("preprocessing_recipe"),
        "preprocessing_search_results": meta.get("preprocessing_search_results"),
        "feature_names": meta.get("feature_names", []),
        "selected_features": meta.get("selected_features", []),
        "preprocessing": safe,
        "dataset_profile": profile,
    }


def get_selection_report(automl: "UnifiedAutoML") -> Dict[str, Any]:
    """Full selection report dict for API consumers."""
    meta = automl.training_metadata or {}
    prep = build_preprocessing_report(automl)
    return {
        "selection": build_selection_decision(automl),
        "preprocessing": {
            "recipe": meta.get("preprocessing_recipe"),
            "search_results": meta.get("preprocessing_search_results"),
            "cv_mean": meta.get("preprocessing_cv_mean"),
            "scoring": meta.get("preprocessing_scoring"),
            "report": prep,
        },
        "leaderboard": automl.get_model_comparison(),
        "holdout_metrics": meta.get("holdout_metrics", {}),
        "regression_metrics": meta.get("regression_metrics", {}),
        "evaluation_results": meta.get("evaluation_results", {}),
        "best_score": automl.best_score,
        "model_type": type(automl.best_model).__name__ if automl.best_model else None,
        "training_time": meta.get("training_time"),
        "reproducibility": {
            "random_state": automl.config.get("random_state", 42),
            "holdout_fraction": 0.2,
            "search_depth": getattr(automl, "search_depth_", None),
            "model_family": meta.get("model_family"),
            "scoring": meta.get("scoring"),
        },
    }


def selection_summary(automl: "UnifiedAutoML") -> str:
    """Short console-friendly selection summary."""
    if not automl.is_fitted:
        return "Model is not fitted."
    decision = build_selection_decision(automl)
    winner = decision["winner"]
    lines = [
        f"Winner: {winner.get('model')} (CV={winner.get('cv_mean', 0):.4f})",
    ]
    if decision.get("runner_up"):
        ru = decision["runner_up"]
        lines.append(f"Runner-up: {ru.get('model')} (CV={ru.get('cv_mean', 0):.4f})")
        if decision.get("margin") is not None:
            lines.append(f"Margin: {decision['margin']:.4f}")
    ens = decision.get("ensemble", {})
    lines.append(f"Ensemble: {ens.get('reason', 'n/a')}")
    holdout = (automl.training_metadata or {}).get("holdout_metrics", {})
    if holdout:
        primary = holdout.get("primary_score") or holdout.get("r2_score") or holdout.get("accuracy")
        if primary is not None:
            lines.append(f"Holdout: {primary:.4f}")
    return "\n".join(lines)


def generate_training_report(automl: "UnifiedAutoML") -> str:
    """Human-readable markdown training report."""
    if not automl.is_fitted:
        return "# AutoForge Training Report\n\nModel is not fitted.\n"

    meta = automl.training_metadata or {}
    decision = build_selection_decision(automl)
    prep = build_preprocessing_report(automl)
    holdout_split = meta.get("holdout_split") or {}
    n_train = holdout_split.get("train_size", "n/a")
    n_holdout = holdout_split.get("holdout_size", "n/a")
    dataset_profile = meta.get("dataset_profile") or {}
    missing_pct = dataset_profile.get("missing_pct")
    lines = [
        "# AutoForge Training Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Dataset",
        f"- Task: {meta.get('task_type', 'unknown')}",
        f"- Data type: {meta.get('data_type', 'unknown')}",
        f"- Features selected: {meta.get('n_features_selected', 'n/a')}",
        f"- Train / holdout rows: {n_train} / {n_holdout}",
        f"- Feature selection on train only: {meta.get('feature_selection_on_train_only', True)}",
    ]
    if missing_pct is not None:
        lines.append(f"- Missing values: {missing_pct:.2f}%")
    lines.extend([
        "",
        "## Preprocessing",
        f"- Recipe: {meta.get('preprocessing_recipe', prep.get('preprocessing', {}).get('recipe_name', 'default'))}",
        f"- Pipeline: {prep.get('preprocessing', {}).get('pipeline_type', 'legacy')}",
        f"- Scale: {prep.get('preprocessing', {}).get('scale_type', prep.get('preprocessing', {}).get('scale_features', 'n/a'))}",
    ])
    search_results = meta.get("preprocessing_search_results") or []
    if search_results:
        lines.extend(["", "### Preprocessing search (CV screen)", ""])
        sorted_rows = sorted(
            [r for r in search_results if "composite_score" in r],
            key=lambda r: r.get("composite_score", 0),
            reverse=True,
        )
        for i, row in enumerate(sorted_rows):
            tag = " **selected**" if row.get("selected") or row.get("recipe") == meta.get("preprocessing_recipe") else ""
            delta = row.get("delta_vs_winner")
            delta_txt = f" (Δ vs winner: {delta:.4f})" if delta is not None else ""
            if row.get("margin_over_runner_up") is not None:
                delta_txt = f" (margin: {row['margin_over_runner_up']:.4f})"
            lines.append(
                f"- {row.get('recipe')}: {row.get('cv_mean', 0):.4f} ± {row.get('cv_std', 0):.4f}"
                f"{tag}{delta_txt}"
            )
        for row in search_results:
            if "error" in row:
                lines.append(f"- {row.get('recipe')}: failed ({row['error']})")
    lines.extend([
        "",
        "## Leaderboard (CV)",
        "",
        "| Model | CV Mean | CV Std | Train (s) | Holdout | Trials | Backend |",
        "|-------|---------|--------|-----------|---------|--------|---------|",
    ])
    for row in automl.get_model_comparison():
        name = str(row.get("model", ""))
        if name.startswith("*"):
            continue
        trials = row.get("trials_run", "—")
        train_t = row.get("train_time")
        train_txt = f"{train_t:.2f}" if isinstance(train_t, (int, float)) else "—"
        holdout = row.get("holdout")
        holdout_txt = f"{holdout:.4f}" if isinstance(holdout, (int, float)) else "—"
        early = " (early-stop)" if row.get("early_stopped") else ""
        lines.append(
            f"| {name}{early} | {row.get('cv_mean', 0):.4f} | {row.get('cv_std', 0):.4f} | "
            f"{train_txt} | {holdout_txt} | {trials} | {row.get('backend', '')} |"
        )
    winner = decision["winner"]
    lines.extend([
        "",
        "## Selected model",
        f"- **{winner.get('model')}**",
        f"- CV score: {winner.get('cv_mean', 0):.4f}",
        f"- Params: `{winner.get('best_params', {})}`",
        "",
        "## Ensemble decision",
        f"- {decision['ensemble'].get('reason', 'n/a')}",
    ])
    ens_info = meta.get("ensemble_info") or {}
    if ens_info.get("member_scores"):
        lines.append("- Members:")
        for name, score in ens_info["member_scores"].items():
            lines.append(f"  - {name}: CV={score:.4f}")
    if ens_info.get("ensemble_score") is not None:
        lines.append(f"- Ensemble CV score: {ens_info['ensemble_score']:.4f}")
    if ens_info.get("single_model_score") is not None:
        lines.append(f"- Best single-model CV: {ens_info['single_model_score']:.4f}")
    lines.extend([
        "",
        "## Holdout metrics (20%)",
    ])
    holdout = meta.get("holdout_metrics", {})
    if holdout:
        for key, val in sorted(holdout.items()):
            if isinstance(val, (int, float)) and not key.startswith("_"):
                lines.append(f"- {key}: {val:.4f}")
    else:
        lines.append("- No holdout metrics recorded.")
    reg = meta.get("regression_metrics", {})
    if reg:
        lines.extend(["", "## Regression (CV + holdout)"])
        for key, val in sorted(reg.items()):
            if isinstance(val, (int, float)):
                lines.append(f"- {key}: {val:.4f}")
    lines.extend([
        "",
        "## Reproducibility",
        f"- Random state: {automl.config.get('random_state', 42)}",
        f"- Search depth: {getattr(automl, 'search_depth_', 'n/a')}",
        f"- Scoring: {meta.get('scoring', 'n/a')}",
        f"- Model family: {meta.get('model_family', 'ml')}",
        "",
        "## Warnings",
    ])
    warnings_list = []
    if meta.get("task_type") == "classification":
        warnings_list.append("Verify class balance for imbalanced datasets.")
    if meta.get("feature_selection_ratio", 1.0) < 0.5:
        warnings_list.append("Aggressive feature selection applied.")
    if not warnings_list:
        warnings_list.append("No critical warnings recorded.")
    for w in warnings_list:
        lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


def save_report_bundle(
    model_dir: Path,
    automl: "UnifiedAutoML",
) -> Dict[str, str]:
    """Write REPORT.md, leaderboard.csv, selection_decision.json, preprocessing_report.json."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    report_md = generate_training_report(automl)
    report_path = model_dir / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")

    rows = automl.get_model_comparison()
    lb_path = model_dir / "leaderboard.csv"
    pd.DataFrame(rows).to_csv(lb_path, index=False)

    decision_path = model_dir / "selection_decision.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(build_selection_decision(automl)), f, indent=2)

    prep_path = model_dir / "preprocessing_report.json"
    with open(prep_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(build_preprocessing_report(automl)), f, indent=2)

    return {
        "REPORT.md": str(report_path),
        "leaderboard.csv": str(lb_path),
        "selection_decision.json": str(decision_path),
        "preprocessing_report.json": str(prep_path),
    }
