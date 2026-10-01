import pandas as pd
import numpy as np
import pathlib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
FEATURES_DATA_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_FILE = FEATURES_DATA_DIR / "laliga_features.csv"

df = pd.read_csv(FEATURES_FILE)


def train_model(df: pd.DataFrame):
    """Train a logistic regression model on the provided DataFrame."""
    train_seasons = [
        "2013/14",
        "2014/15",
        "2015/16",
        "2016/17",
        "2017/18",
        "2018/19",
        "2019/20",
        "2020/21",
    ]
    val_seasons = ["2021/22", "2022/23"]
    test_seasons = ["2023/24", "2024/25", "2025/26"]

    df_train = df[df["Season"].isin(train_seasons)]
    df_val = df[df["Season"].isin(val_seasons)]
    df_test = df[df["Season"].isin(test_seasons)]

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    # print(df.columns.tolist())

    feature_columns = [
        "market_movement_H",
        "market_movement_D",
        "market_movement_A",
        "ELO_home",
        "ELO_away",
        "ELO_diff",
        "HomeTeam_goals_scored_last3_matches",
        "HomeTeam_goals_conceded_last3_matches",
        "HomeTeam_shots_on_target_last3_matches",
        "HomeTeam_shots_on_target_conceded_last3_matches",
        "HomeTeam_corners_last3_matches",
        "HomeTeam_corners_conceded_last3_matches",
        "HomeTeam_fouls_last3_matches",
        "HomeTeam_fouls_conceded_last3_matches",
        "HomeTeam_points_last3_matches",
        "HomeTeam_goals_scored_last5_matches",
        "HomeTeam_goals_conceded_last5_matches",
        "HomeTeam_shots_on_target_last5_matches",
        "HomeTeam_shots_on_target_conceded_last5_matches",
        "HomeTeam_corners_last5_matches",
        "HomeTeam_corners_conceded_last5_matches",
        "HomeTeam_fouls_last5_matches",
        "HomeTeam_fouls_conceded_last5_matches",
        "HomeTeam_points_last5_matches",
        "HomeTeam_goals_scored_last7_matches",
        "HomeTeam_goals_conceded_last7_matches",
        "HomeTeam_shots_on_target_last7_matches",
        "HomeTeam_shots_on_target_conceded_last7_matches",
        "HomeTeam_corners_last7_matches",
        "HomeTeam_corners_conceded_last7_matches",
        "HomeTeam_fouls_last7_matches",
        "HomeTeam_fouls_conceded_last7_matches",
        "HomeTeam_points_last7_matches",
        "HomeTeam_goals_scored_last10_matches",
        "HomeTeam_goals_conceded_last10_matches",
        "HomeTeam_shots_on_target_last10_matches",
        "HomeTeam_shots_on_target_conceded_last10_matches",
        "HomeTeam_corners_last10_matches",
        "HomeTeam_corners_conceded_last10_matches",
        "HomeTeam_fouls_last10_matches",
        "HomeTeam_fouls_conceded_last10_matches",
        "HomeTeam_points_last10_matches",
        "AwayTeam_goals_scored_last3_matches",
        "AwayTeam_goals_conceded_last3_matches",
        "AwayTeam_shots_on_target_last3_matches",
        "AwayTeam_shots_on_target_conceded_last3_matches",
        "AwayTeam_corners_last3_matches",
        "AwayTeam_corners_conceded_last3_matches",
        "AwayTeam_fouls_last3_matches",
        "AwayTeam_fouls_conceded_last3_matches",
        "AwayTeam_points_last3_matches",
        "AwayTeam_goals_scored_last5_matches",
        "AwayTeam_goals_conceded_last5_matches",
        "AwayTeam_shots_on_target_last5_matches",
        "AwayTeam_shots_on_target_conceded_last5_matches",
        "AwayTeam_corners_last5_matches",
        "AwayTeam_corners_conceded_last5_matches",
        "AwayTeam_fouls_last5_matches",
        "AwayTeam_fouls_conceded_last5_matches",
        "AwayTeam_points_last5_matches",
        "AwayTeam_goals_scored_last7_matches",
        "AwayTeam_goals_conceded_last7_matches",
        "AwayTeam_shots_on_target_last7_matches",
        "AwayTeam_shots_on_target_conceded_last7_matches",
        "AwayTeam_corners_last7_matches",
        "AwayTeam_corners_conceded_last7_matches",
        "AwayTeam_fouls_last7_matches",
        "AwayTeam_fouls_conceded_last7_matches",
        "AwayTeam_points_last7_matches",
        "AwayTeam_goals_scored_last10_matches",
        "AwayTeam_goals_conceded_last10_matches",
        "AwayTeam_shots_on_target_last10_matches",
        "AwayTeam_shots_on_target_conceded_last10_matches",
        "AwayTeam_corners_last10_matches",
        "AwayTeam_corners_conceded_last10_matches",
        "AwayTeam_fouls_last10_matches",
        "AwayTeam_fouls_conceded_last10_matches",
        "AwayTeam_points_last10_matches",
    ]

    X_train_nan = df_train[feature_columns]
    y_train = df_train["FTR"]
    means = X_train_nan.mean()
    X_train = X_train_nan.fillna(means)

    X_val_nan = df_val[feature_columns]
    y_val = df_val["FTR"]
    means = X_train.mean()
    X_val = X_val_nan.fillna(means)

    X_test_nan = df_test[feature_columns]
    y_test = df_test["FTR"]
    means = X_train.mean()
    X_test = X_test_nan.fillna(means)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


def logistic_regression_model(X_train, y_train, X_val, y_val, X_test, y_test):
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
    logistic_regression.fit(X_train, y_train)
    logistic_regression_val_pred = logistic_regression.predict_proba(X_val)
    logistic_regression_test_pred = logistic_regression.predict_proba(X_test)
    logistic_regression_classes = logistic_regression.classes_

    print("LogisticRegression Model Results:")
    print(f"Validation Shape: {logistic_regression_val_pred.shape}")
    print(f"Test Shape: {logistic_regression_test_pred.shape}")
    print(f"Classes: {logistic_regression_classes}")

    return (
        logistic_regression_val_pred,
        logistic_regression_test_pred,
        logistic_regression_classes,
    )


def evaluate_model(y_true, y_pred):
    """Log Loss for the model predictions."""
    log_loss_result = log_loss(y_true, y_pred)

    """Brier Score for the model predictions."""
    result_H = (y_true == "H").astype(int)
    result_D = (y_true == "D").astype(int)
    result_A = (y_true == "A").astype(int)

    y_true_bs = np.column_stack((result_A, result_D, result_H))
    brier_score_result = np.mean(
        [brier_score_loss(y_true_bs[:, i], y_pred[:, i]) for i in range(3)]
    )

    """Accuracy for the model predictions."""
    predicted_result = np.argmax(y_pred, axis=1)
    actual_result = np.argmax(y_true_bs, axis=1)
    accuracy_result = (predicted_result == actual_result).mean()

    print(f"Log Loss: {log_loss_result:.4f}")
    print(f"Brier Score: {brier_score_result:.4f}")
    print(f"Accuracy: {accuracy_result:.4f}")

    return log_loss_result, brier_score_result, accuracy_result


def save_metrics_to_mlflow(log_loss_result, brier_score_result, accuracy_result):
    with mlflow.start_run():
        """Save the evaluation metrics to MLflow."""
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_metric("log_loss", log_loss_result)
        mlflow.log_metric("brier_score", brier_score_result)
        mlflow.log_metric("accuracy", accuracy_result)


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = train_model(df)
    (
        logistic_regression_val_pred,
        logistic_regression_test_pred,
        logistic_regression_classes,
    ) = logistic_regression_model(X_train, y_train, X_val, y_val, X_test, y_test)

    log_loss_val, brier_score_val, accuracy_val = evaluate_model(
        y_val, logistic_regression_val_pred
    )
    log_loss_test, brier_score_test, accuracy_test = evaluate_model(
        y_test, logistic_regression_test_pred
    )

    save_metrics_to_mlflow(log_loss_val, brier_score_val, accuracy_val)
    save_metrics_to_mlflow(log_loss_test, brier_score_test, accuracy_test)


if __name__ == "__main__":
    main()
