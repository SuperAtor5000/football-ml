import json
import pathlib
import sys

import mlflow.xgboost

from src.models.predict import CLASSES, load_model

EXPORT_DIR = pathlib.Path("models/final")


def export_model(run_id: str) -> None:
    """Save the model of an MLflow run (and its class order) into a plain folder."""
    model = load_model(run_id)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.xgboost.save_model(model, str(EXPORT_DIR / "model"))

    with open(EXPORT_DIR / "classes.json", "w") as f:
        json.dump({"classes": CLASSES}, f)


if __name__ == "__main__":
    export_model(sys.argv[1])
