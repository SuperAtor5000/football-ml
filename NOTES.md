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
- Single val (2021/22-2022/23): 0.9921 without vs 0.9917 with market (0.0004, noise).
- Walk-forward, 5 folds (2018/19-2022/23), diff = with market - without market:
  -0.0018, -0.0021, -0.0019, +0.0018, -0.0019. Mean -0.0012,
  paired bootstrap 95% CI (-0.0029, +0.0005). Crosses 0: not significant.
- With market wins 4 of 5 folds by ~0.002, even though the params were tuned
  WITHOUT market (which favors the no-market variant). Small, consistent but
  statistically unclear signal. The 2021/22 reversal is most likely noise.
- Decision: exclude it by design (needs closing odds, only available minutes before
  kickoff), accepting a possible ~0.001 Log Loss cost. Not because it "adds nothing".

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
- Margin of victory (multiplier on k, normalized by its mean over train so the
  average k is unchanged). 9 shapes tested: none, linear, logarithmic, square_root,
  world_football_elo, harmonic, harmonic_plus_one, saturated, capped.
  Val log loss (one-feature logistic regression, k=8): none 0.9957, all others
  0.9958-0.9968. Grid over k in {2,...,16}: best per variant 0.9956 (none) to 0.9967
  (linear). No variant beats "none" (differences < 0.002, i.e. noise).
- Conclusion: with this method, goal difference in the ELO adds nothing measurable.
  Default stays "none"; the code is kept. Likely reasons (hypotheses): goals are
  noisy, k is already low, and a multiplier is a patch (a real fix would model the
  expected goal margin, e.g. Poisson / Dixon-Coles).
- Normalization scales on train (mean G): none 1.00, saturated 1.21, square_root 1.22,
  world_football_elo 1.23, harmonic 1.25, logarithmic 1.33, capped 1.51,
  linear 1.60, harmonic_plus_one 1.99. Without normalizing, harmonic_plus_one would
  have doubled the effective k.

  ## Standings features
- Added per team (home/away): points accumulated before the match, distance in points
  to the 7th place (Europe) and to the 18th place (relegation). Computed per date, only
  from previous matches. No tie-breaks needed (only points matter).
- Leakage audit: mutation test (changing a match result does not change that match's
  features) passed.
- Walk-forward ablation (diff = base - variant, positive = variant better):
  acc_only +0.0011 (CI -0.0003, +0.0023), acc+dist +0.0019 (CI +0.0001, +0.0038),
  remaining_only +0.0007, all8 +0.0023 (CI +0.0003, +0.0042).
- Decision: adopt acc+dist (6 columns). points_remaining adds ~0.0004 (noise).
- Caveats: CI barely excludes 0; 2022/23 goes against in most variants; several
  comparisons on the same folds.
- Model seed (random_state) moves Optuna's best val log loss by ~0.001, the same size
  as the improvements measured. Pending: repeat walk-forward with 3 seeds.
- Retuned XGBoost val log loss: 0.9934 (same as before, within noise).

  ## Validation system
- Walk-forward: 5 folds, train always the past (starting 2015/16, warmup excluded),
  validate on the next season (2018/19 ... 2022/23). Test untouched.
- Baseline (no market, tuned params): per-season Log Loss 1.0248, 1.0125, 0.9947,
  0.9973, 0.9866 (mean 1.0032). Range across seasons ~0.038, so a single-season val
  cannot detect improvements of 0.001-0.005.
- Early folds are worse partly because of less training data and harder seasons;
  the two can't be separated.
- Paired bootstrap on per-match loss differences (2000 resamples, with replacement).
  Verified with synthetic data (constant diff -0.1 -> CI (-0.1, -0.1)).
- Decision rule for a new feature: wins in >= 4 of 5 folds, relevant mean gain,
  bootstrap CI does not cross 0. Ties go to the simpler variant.
  Caveat: with many ideas tested, some pass by chance; test stays the final check.

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
- My a-priori estimates (e.g. mean margin scales, expected gain from goal margin) were
  wrong; measure instead of guessing.
- A feature like goal_diff = FTHG - FTAG exists only after the match: using it as input
  is leakage. Past goal difference must come via rolling features or the ELO update.

## To do
1. Cheap ideas, each measured with walk-forward + paired bootstrap: fewer rolling
   windows, matchday, no_crowd, home/away rolling. Then freeze the feature set.
2. Opponent strength, home/away split, matchday, no_crowd, end-of-season handicap.
3. Refactor: read the CSV inside main (not at module level in train.py), shared season
   lists in one module, propagate elo_params to build_features.
4. Optuna + MLflow nested runs, FastAPI + Docker, Prefect, Streamlit.
5. goals model (Poisson / Dixon-Coles), opponent strength, end-of-season handicap.