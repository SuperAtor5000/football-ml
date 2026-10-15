# Project notes: La Liga match outcome prediction

## Data and splits
- 4940 matches, 13 seasons (2013/14 to 2025/26), source: football-data.co.uk.
- Train 2013/14-2020/21 (3040), val 2021/22-2022/23 (760), test 2023/24-2025/26 (1140).
- Temporal split only, never random.
- Pinnacle has no closing odds for 192 matches of 2025/26. The gap is in the source
  (checked: the current website has the same empty values).

## Bookmaker references (normalized closing odds)
- B365 closing: val 0.9842, test 0.9544 (1140 matches, clean reference).
- Pinnacle closing: test 0.9485 on its 948 available matches.
- Bookmakers differ by < 0.005 on common matches (noise).
- Opening vs closing odds: only ~0.001-0.005 Log Loss difference.
- Val is harder than test for ALL models and bookmakers (confirmed with bookmakers,
  which do not depend on my model). Never compare a val number with a test number.

## Models (Log Loss)
| Model | Val | Test |
|---|---|---|
| Logistic regression, old ELO, with market | 1.0192 | 0.9857 |
| XGBoost, default params | 1.2773 | 1.1978 |
| XGBoost tuned, old ELO, with market | 0.9917 | 0.9714 |
| XGBoost tuned, new ELO, no market, with warmup (final) | 0.9934 | 0.9750 |

- Default XGBoost overfits (3040 rows, ~75 features); Optuna fixes it.
- Good region: max_depth 2-3, strong regularization (reg_lambda ~50-100),
  learning_rate x n_estimators ~ 2-3.
- Differences between the tuned variants (0.9917-0.9958) are within noise.

## market_movement ablation
- Without market: val 0.9921 vs 0.9917 with market. No measurable contribution.
- Decision: exclude it (it also needs closing odds, only available minutes before kickoff).

## ELO
- Tuned with a one-feature logistic regression (ELO_diff) to avoid XGBoost noise.
  Baseline 1.0057 (K=20) -> ~0.9956.
- Frozen values: k=8, home_advantage=65, mean_reversion=0, promoted teams=1300.
- Positive mean reversion hurts (0 -> 0.1: +0.005). Negative gave -0.001 (noise,
  no physical meaning), discarded.
- home_advantage barely matters (the regression intercept absorbs it); the 140-200
  values Optuna returned are noise.
- ELO needs ~2 seasons to separate teams (std 25 -> 107 -> ~130).
  Warmup: 2013/14 and 2014/15 are used to compute ELO but not for training.
- With XGBoost the new ELO did NOT measurably improve results (it did with the
  one-feature regression).

## Gap vs the market
- Test: ~0.975 (model) vs ~0.954 (B365 closing): ~0.02.
- Val has been used several times to choose parameters, so its numbers are optimistic.
- Test has been looked at several times; from now on it is only reported.

## Mistakes and lessons
- sklearn log_loss ignores the order of `labels` and assumes A, D, H. Columns in
  another order give absurd values (>1.09) with no error. Sanity check: Log Loss above
  ln(3) = 1.0986 is impossible for a bookmaker.
- EDA baselines mixed train+val+test: not comparable with test.
- download.py depends on today's date (it downloaded 2026/27): pin the last season.

## To do
1. Walk-forward validation (5 folds) + paired bootstrap to measure real improvements.
2. ELO with goal difference.
3. Rolling windows: try removing windows, exponential weighting.
4. Opponent strength, home/away split, matchday, no_crowd, end-of-season handicap.
5. Refactor: read the CSV inside main (not at module level in train.py), shared season
   lists in one module, propagate elo_params to build_features.
6. Optuna + MLflow nested runs, FastAPI + Docker, Prefect, Streamlit.