# AutoForge Classical Benchmarks

Generated: 2026-05-26T14:26:18.965918+00:00

## Exit criteria

- Balanced beats fast (holdout): **7/7**
- Balanced within 5% of sklearn baseline: **6/7**
- Predict round-trip OK: **True**

## Summary

| Dataset | Task | System | Holdout | CV | Time (s) | Recipe | Report | RT | Winner |
|---------|------|--------|---------|-----|----------|--------|--------|-----|--------|
| iris | classification | sklearn_baseline | 0.9000 | — | 0.41 | sklearn_rf_pipeline | no | ok | RandomForest |
| iris | classification | autoforge_fast | 0.9333 | 0.9667 | 35.14 | — | yes | ok | xgboost |
| iris | classification | autoforge_balanced | 0.9667 | 1.0000 | 21.87 | — | yes | ok | xgboost |
| wine | classification | sklearn_baseline | 1.0000 | — | 0.45 | sklearn_rf_pipeline | no | ok | RandomForest |
| wine | classification | autoforge_fast | 0.9722 | 0.9860 | 11.01 | — | yes | ok | logistic_regression |
| wine | classification | autoforge_balanced | 1.0000 | 1.0000 | 9.38 | — | yes | ok | extra_trees |
| california_housing | regression | sklearn_baseline | 0.8053 | — | 18.86 | sklearn_rf_pipeline | no | ok | RandomForest |
| california_housing | regression | autoforge_fast | 0.8305 | -0.2327 | 87.68 | — | yes | ok | lasso |
| california_housing | regression | autoforge_balanced | 0.8305 | -0.2327 | 44.57 | — | yes | ok | lasso |
| regression_synthetic | regression | sklearn_baseline | 0.7946 | — | 0.44 | sklearn_rf_pipeline | no | ok | RandomForest |
| regression_synthetic | regression | autoforge_fast | 0.9999 | -1.5636 | 3.31 | — | yes | ok | svr |
| regression_synthetic | regression | autoforge_balanced | 0.9999 | -1.5636 | 1.89 | — | yes | ok | svr |
| imbalanced_classification | classification | sklearn_baseline | 0.9125 | — | 0.37 | sklearn_rf_pipeline | no | ok | RandomForest |
| imbalanced_classification | classification | autoforge_fast | 0.8875 | 0.8812 | 4.55 | — | yes | ok | xgboost |
| imbalanced_classification | classification | autoforge_balanced | 0.9000 | 1.0000 | 6.55 | — | yes | ok | xgboost |
| high_cardinality_cats | classification | sklearn_baseline | 0.9000 | — | 0.24 | sklearn_rf_pipeline | no | ok | RandomForest |
| high_cardinality_cats | classification | autoforge_fast | 0.9300 | 0.9525 | 2.59 | — | yes | ok | logistic_regression |
| high_cardinality_cats | classification | autoforge_balanced | 0.9300 | 0.9525 | 3.95 | — | yes | ok | logistic_regression |
| housing_csv | regression | sklearn_baseline | 0.6109 | — | 0.45 | sklearn_rf_pipeline | no | ok | RandomForest |
| housing_csv | regression | autoforge_fast | 0.0000 | -1113988579365.4512 | 2.37 | — | yes | ok | svr |
| housing_csv | regression | autoforge_balanced | 0.0000 | -1113988579365.4512 | 1.95 | — | yes | ok | svr |

## Notes

- Holdout: AutoForge internal 20% split (feature selection on train only).
- Baseline: sklearn Pipeline (impute + scale/ordinal + RandomForest).
- RT: save_model → load_model → predict round-trip.
- Re-run: `python -m benchmarking.classical_suite`
