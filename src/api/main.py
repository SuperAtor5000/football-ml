from collections import defaultdict
from contextlib import asynccontextmanager
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.predict import (
    build_prediction_rows,
    probabilities_to_dicts,
)

MODEL_DIR = "models/final/model"
HISTORY_FILE = "data/processed/laliga.csv"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup
    state["history"] = pd.read_csv(HISTORY_FILE)
    state["model"] = mlflow.xgboost.load_model(MODEL_DIR)
    yield


app = FastAPI(title="La Liga 1X2 predictor", lifespan=lifespan)


class Match(BaseModel):
    home: str
    away: str
    date: str  # YYYY-MM-DD


class PredictRequest(BaseModel):
    matches: list[Match]


@app.post("/predict")
def predict(request: PredictRequest):
    # 1. Group matches by date, remembering the original position of each one
    by_date = defaultdict(list)
    for position, match in enumerate(request.matches):
        by_date[match.date].append((position, match))

    results = [None] * len(request.matches)

    # 2. One feature build per date
    for date, items in by_date.items():
        pairs = [(m.home, m.away) for _, m in items]
        try:
            rows = build_prediction_rows(state["history"], pairs, date=date)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # 3. Probabilities with the model loaded at startup
        probs = probabilities_to_dicts(state["model"], rows)

        # 4. Put each result back in its original position
        for (position, m), p in zip(items, probs):
            results[position] = {"home": m.home, "away": m.away, "date": m.date, **p}

    return {"predictions": results}


@app.get("/health")
def health():
    return {"status": "ok"}
