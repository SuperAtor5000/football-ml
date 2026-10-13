from train import train_model_xgboost, df
import optuna
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded, encoder = (
    train_model_xgboost(df)
)


def objective(trial):
    # Define the hyperparameter search space
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 800),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 50, log=True),
    }

    # Create and train the XGBoost model
    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train_encoded)

    # Make predictions on the validation set
    y_val_pred_proba = model.predict_proba(X_val)

    # Calculate evaluation metrics
    log_loss_val = log_loss(
        y_val_encoded, y_val_pred_proba, labels=encoder.transform(encoder.classes_)
    )

    return log_loss_val


study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=100)

print("Best hyperparameters: ", study.best_params)
print("Best log loss: ", study.best_value)
