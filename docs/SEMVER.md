# Semantic Versioning Policy

AutoForge follows [SemVer 2.0.0](https://semver.org/).

## Version format

`MAJOR.MINOR.PATCH` (e.g. `1.0.0`)

## When to bump

| Bump | When |
|------|------|
| **MAJOR** | Breaking changes to the public sklearn API (`AutoForgeClassifier` / `AutoForgeRegressor`), saved model bundle layout, or default training behavior that changes predictions on the same data/config |
| **MINOR** | Backward-compatible features: new models, recipes, metrics, CLI flags, or serving endpoints |
| **PATCH** | Bug fixes, performance improvements, documentation, and internal refactors with no API impact |

## Public API surface

- `autoforge` package and `core.estimator` sklearn wrappers
- `UnifiedAutoML.fit` / `predict` / `save_model` / `load_model`
- Saved model directory layout (`model/`, `training.json`, `REPORT.md`, etc.)
- CLI (`main.py` / `autoforge` script)

Experimental modules (NAS, multimodal, DL) may change in minor releases until promoted to stable.

## Pre-1.0 note

Version `1.0.0` marks the classical tabular AutoML golden path as stable. Subsequent breaking changes require a major bump.
