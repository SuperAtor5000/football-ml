## Results

Evaluated on 1140 held-out matches (seasons 2023/24 to 2025/26), metric: log loss
(lower is better).

| Model | Log loss |
|---|---|
| Uniform probabilities (1/3 each) | 1.0986 |
| **XGBoost (final model)** | **0.9739** |
| Bet365 closing odds (reference) | 0.9544 |

The model learns real signal but stays ~0.02 behind the market, which aggregates
information I do not have (lineups, injuries, betting volume). The model does not use
odds as input.

## Architecture
- **Data**: 13 seasons (2013/14 to 2025/26) plus the current one for serving.
- **Features (81)**: ELO ratings, rolling form over 3/5/7/10 matches, and points
  accumulated / distance to the European and relegation places.
- **Model**: XGBoost, heavily regularized (depth 2), tuned with Optuna.
- **Tracking**: MLflow logs parameters, feature list, class order, metrics and model.
- **Serving**: FastAPI loads the model and history once at startup.

## Engineering decisions

- **Temporal splits only** (train 2013/14-2020/21, validation 2021/22-2022/23, test
  afterwards). ELO warmup seasons are used to compute ratings but not for training.
- **Walk-forward validation + paired bootstrap** to decide which features to keep.
  Single-season validation was too noisy (~0.04 spread between seasons) to detect
  improvements of 0.002. Ideas that failed the test were discarded and documented
  (market movement, margin-of-victory ELO, COVID attendance, home/away form).
- **Same code for training and serving**: the API builds features for future matches
  with the exact function used for training. A test reconstructs a played match as if
  it were in the future and checks that the features are identical.
- **Leakage audits** (mutation test) and sanity checks (a silent class-order bug in
  log loss produced plausible but wrong numbers).

Full experiment log, results and mistakes: see [NOTES.md](NOTES.md).

## Run it

Requirements: Python 3.11 + [uv](https://docs.astral.sh/uv/), or Docker.

```bash
# API with Docker
docker build -t laliga-api .
docker run -p 8000:8000 laliga-api
```

Open http://127.0.0.1:8000/docs and call `POST /predict`:

```json
{"matches": [{"home": "Barcelona", "away": "Alaves", "date": "2026-10-31"}]}
```

Team names must match the dataset (e.g. `Ath Madrid`, `Espanol`, `Sociedad`).
Matches already in the history are rejected (HTTP 422).

Local training:

```bash
uv sync
uv run python -m src.models.train
uv run mlflow ui
```

## Data

Source: [football-data.co.uk](https://www.football-data.co.uk). <!-- TODO: confirm the
site's terms of use before keeping the processed CSVs in the repo. -->
Data is not refreshed automatically: run `download`, `process` and `build_features`,
then restart the API.

## Status and next steps

Done: data pipeline, features, tuning, MLflow, FastAPI, Docker.

Planned: automated data refresh (Prefect), automated tests (pytest), Streamlit front
end, a goals-based model (Poisson / Dixon-Coles).

Known limitations: the history is not updated automatically, each request recomputes
features (a few seconds), and teams absent the previous season are all treated as
promoted for ELO purposes.