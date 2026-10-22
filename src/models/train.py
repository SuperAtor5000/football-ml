import pandas as pd
import numpy as np
import pathlib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from xgboost import XGBClassifier

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
FEATURES_DATA_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_FILE = FEATURES_DATA_DIR / "laliga_features.csv"


BEST_XGB_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.005861421069352081,
    "n_estimators": 793,
    "min_child_weight": 8,
    "subsample": 0.9722666745444034,
    "colsample_bytree": 0.4830901143831626,
    "reg_lambda": 92.56287040362311,
}

STANDINGS_FEATURES = [
    f"{side}_{name}"
    for side in ["HomeTeam", "AwayTeam"]
    for name in [
        "points_accumulated_before_match",
        "dist_europe",
        "dist_relegation",
    ]
]

df = pd.read_csv(FEATURES_FILE)


def train_model_logistic_regression(
    df: pd.DataFrame, include_market, warmup_seasons=()
):
    """Train a logistic regression model on the provided DataFrame."""
    df_train, df_val, df_test = divide_df_train_val_test(df, warmup_seasons)
    feature_columns = select_features(include_market)

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    # print(df.columns.tolist())

    X_train_nan, y_train = split_X_y(df_train, feature_columns)
    means = X_train_nan.mean()
    X_train = X_train_nan.fillna(means)

    X_val_nan, y_val = split_X_y(df_val, feature_columns)
    means = X_train.mean()
    X_val = X_val_nan.fillna(means)

    X_test_nan, y_test = split_X_y(df_test, feature_columns)
    means = X_train.mean()
    X_test = X_test_nan.fillna(means)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


def train_model_xgboost(df: pd.DataFrame, include_market, warmup_seasons=()):
    """Train a XGBoost model on the provided DataFrame."""
    df_train, df_val, df_test = divide_df_train_val_test(df, warmup_seasons)
    feature_columns = select_features(include_market)

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    # print(df.columns.tolist())

    X_train, y_train = split_X_y(df_train, feature_columns)
    X_val, y_val = split_X_y(df_val, feature_columns)
    X_test, y_test = split_X_y(df_test, feature_columns)

    """Encode the target variable (y) using LabelEncoder to convert categorical labels into numerical format for XGBoost."""
    encoder = LabelEncoder()
    # Fit the encoder on the training labels (a match could finish as a win, draw, or loss so it finds all the possible results in the dataset) and transform them (f.e. "A"=0, "D"=1, "H"=2)
    y_train_encoded = encoder.fit_transform(y_train)
    # Keep the same encoder for validation and test sets (f.e. "A"=0, "D"=1, "H"=2) to ensure consistent label encoding
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)

    print(encoder.classes_)

    return (
        X_train,
        y_train_encoded,
        X_val,
        y_val_encoded,
        X_test,
        y_test_encoded,
        encoder,
    )


def logistic_regression_model(X_train, y_train, X_val, X_test):
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
    logistic_regression.fit(X_train, y_train)
    logistic_regression_val_pred = logistic_regression.predict_proba(X_val)
    logistic_regression_test_pred = logistic_regression.predict_proba(X_test)
    logistic_regression_classes = logistic_regression.classes_

    print("LogisticRegression Model Results:")
    print(f"        Validation Shape: {logistic_regression_val_pred.shape}")
    print(f"        Test Shape: {logistic_regression_test_pred.shape}")
    print(f"        Classes: {logistic_regression_classes}")

    return (
        logistic_regression_val_pred,
        logistic_regression_test_pred,
        logistic_regression_classes,
    )


def xgboost_model(X_train, y_train, X_val, X_test, params):
    xgb_model = XGBClassifier(**params, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_val_xgb_pred = xgb_model.predict_proba(X_val)
    y_test_xgb_pred = xgb_model.predict_proba(X_test)

    print("XGBoost Model Results:")
    print(f"        Validation Shape: {y_val_xgb_pred.shape}")
    print(f"        Test Shape: {y_test_xgb_pred.shape}")

    return y_val_xgb_pred, y_test_xgb_pred


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

    print(f"        Log Loss: {log_loss_result:.4f}")
    print(f"        Brier Score: {brier_score_result:.4f}")
    print(f"        Accuracy: {accuracy_result:.4f}")

    return log_loss_result, brier_score_result, accuracy_result


def save_metrics_to_mlflow(
    log_loss_val,
    brier_score_val,
    accuracy_val,
    log_loss_test,
    brier_score_test,
    accuracy_test,
    model_type,
    include_market,
    warmup_seasons=(),
):
    with mlflow.start_run():
        """Save the evaluation metrics to MLflow."""
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("include_market", include_market)
        mlflow.log_param("warmup_seasons", str(warmup_seasons))
        mlflow.log_metric("log_loss_val", log_loss_val)
        mlflow.log_metric("brier_score_val", brier_score_val)
        mlflow.log_metric("accuracy_val", accuracy_val)
        mlflow.log_metric("log_loss_test", log_loss_test)
        mlflow.log_metric("brier_score_test", brier_score_test)
        mlflow.log_metric("accuracy_test", accuracy_test)


def divide_df_train_val_test(df: pd.DataFrame, warmup_seasons=()):
    """Divide the DataFrame into train, validation, and test sets based on seasons."""
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

    train_seasons = [s for s in train_seasons if s not in warmup_seasons]

    df_train = df[df["Season"].isin(train_seasons)]
    df_val = df[df["Season"].isin(val_seasons)]
    df_test = df[df["Season"].isin(test_seasons)]

    return df_train, df_val, df_test


def select_features(include_market=True, include_standings=True):
    """Select the features to be used for training the model."""
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
    if not include_market:
        feature_columns = [
            c for c in feature_columns if not c.startswith("market_movement")
        ]
    if include_standings:
        feature_columns = feature_columns + STANDINGS_FEATURES
    return feature_columns


def split_X_y(df: pd.DataFrame, feature_columns: list):
    """Divide the DataFrame into features (X) and target (y)."""
    X = df[feature_columns]
    y = df["FTR"]
    return X, y


def main(model_type, include_market, warmup_seasons=()):
    """Main function to train and evaluate models."""

    if model_type == "Logistic Regression":
        """Logistic Regression Model Training and Evaluation"""
        X_train_lr, y_train_lr, X_val_lr, y_val_lr, X_test_lr, y_test_lr = (
            train_model_logistic_regression(df)
        )
        y_val_lr_pred, y_test_lr_pred, lr_classes = logistic_regression_model(
            X_train_lr, y_train_lr, X_val_lr, X_test_lr
        )
        print("    Evaluation Validation Metrics:")
        log_loss_val_lr, brier_score_val_lr, accuracy_val_lr = evaluate_model(
            y_val_lr, y_val_lr_pred
        )
        print("    Evaluation Test Metrics:")
        log_loss_test_lr, brier_score_test_lr, accuracy_test_lr = evaluate_model(
            y_test_lr, y_test_lr_pred
        )
        save_metrics_to_mlflow(
            log_loss_val_lr,
            brier_score_val_lr,
            accuracy_val_lr,
            log_loss_test_lr,
            brier_score_test_lr,
            accuracy_test_lr,
            model_type,
            include_market,
            warmup_seasons,
        )

    elif model_type == "XGBoost Tuned":
        """XGBoost Model Training and Evaluation"""
        (
            X_train_xgb,
            y_train_xgb,
            X_val_xgb,
            y_val_xgb,
            X_test_xgb,
            y_test_xgb,
            encoder,
        ) = train_model_xgboost(df, include_market, warmup_seasons)
        y_val_xgb_pred, y_test_xgb_pred = xgboost_model(
            X_train_xgb, y_train_xgb, X_val_xgb, X_test_xgb, BEST_XGB_PARAMS
        )
        y_val_xgb_decoded = encoder.inverse_transform(y_val_xgb)
        y_test_xgb_decoded = encoder.inverse_transform(y_test_xgb)
        print("    Evaluation Validation Metrics:")
        log_loss_val_xgb, brier_score_val_xgb, accuracy_val_xgb = evaluate_model(
            y_val_xgb_decoded, y_val_xgb_pred
        )
        print("    Evaluation Test Metrics:")
        log_loss_test_xgb, brier_score_test_xgb, accuracy_test_xgb = evaluate_model(
            y_test_xgb_decoded, y_test_xgb_pred
        )
        save_metrics_to_mlflow(
            log_loss_val_xgb,
            brier_score_val_xgb,
            accuracy_val_xgb,
            log_loss_test_xgb,
            brier_score_test_xgb,
            accuracy_test_xgb,
            model_type,
            include_market,
            warmup_seasons,
        )


if __name__ == "__main__":
    ##model_type = "Logistic Regression";
    ##main(model_type)
    model_type = "XGBoost Tuned"
    include_market = (
        False  # Set to True to include market movement features, False otherwise
    )
    for warmup in [(), ("2013/14", "2014/15")]:
        main(model_type, include_market, warmup_seasons=warmup)
