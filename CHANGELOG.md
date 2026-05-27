# Changelog

All notable changes to AutoForge are documented here. Versioning follows [SemVer](https://semver.org/).

## [1.0.0] - 2026-05-26

### Classical tabular AutoML (v1.0)

- Unified preprocessing pipeline with 8 searchable recipes
- Preprocessing search with dual screen models and stability-weighted scoring
- Model search with fair per-model budgets, early-stop, configurable `scoring`
- 20% holdout evaluation; feature selection on train split only
- Saved justification bundle: `REPORT.md`, `leaderboard.csv`, `selection_decision.json`
- Sklearn API and `autoforge` package namespace
- Classical benchmark suite and examples
- CI: fast tests on PR; nightly full suite + benchmarks
- Legacy `api/*` deprecated in favor of `core` / `autoforge`
- `utils/dtype_helpers` for pandas 2/4 compatible categorical detection
- Per-model `train_time` and holdout scores in leaderboard / `REPORT.md`
- `dataset_profile.missing_pct` in training metadata
- Docker + `docker-compose` serving stack; `python -m serving.run`
- Leakage tests (`test_holdout_indices`), recipe round-trips, serve e2e, unseen category
- `docs/SEMVER.md` semver policy
- Classical canon imports (`CLASSICAL_CLASSIFICATION`, `CLASSICAL_REGRESSION`)

### Known limitations

- Deep learning, NAS, and multimodal paths are experimental
- Target encoding uses train-split mean encoding
- No ONNX export in v1.0
- Serving API has no authentication
