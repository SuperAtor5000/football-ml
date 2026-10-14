import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import calculate_elo, PROCESSED_DATA_DIR
from src.models.train import divide_df_train_val_test


def elo_val_log_loss(
    df, k, home_advantage, mean_reversion_factor, initial_elo_promoted
):
    # 1. Calculate ELO ratings with the given hyperparameters
    df_elo = calculate_elo(
        df,
        k=k,
        home_advantage=home_advantage,
        mean_reversion_factor=mean_reversion_factor,
        initial_elo_promoted=initial_elo_promoted,
    )

    # 2. Divide the DataFrame into training, validation, and test sets
    df_train, df_val, _ = divide_df_train_val_test(df_elo)
    df_train = df_train[~df_train["Season"].isin(["2013/14", "2014/15"])]

    # 3. Prepare the features and target for training and validation
    X_train = df_train[["ELO_diff"]]
    y_train = df_train["FTR"]
    X_val = df_val[["ELO_diff"]]
    y_val = df_val["FTR"]

    # 4. Train a simple model (e.g., logistic regression) and evaluate log loss on the validation set
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)
    return log_loss(y_val, proba, labels=model.classes_)


def run_elo_study(df, n_trials=250):
    def objective(trial):
        k = trial.suggest_float("k", 1, 50, log=True)
        home_advantage = trial.suggest_float("home_advantage", 0, 200)
        mean_reversion_factor = trial.suggest_float("mean_reversion_factor", 0, 0.5)
        initial_elo_promoted = trial.suggest_float("initial_elo_promoted", 1150, 1500)

        return elo_val_log_loss(
            df,
            k=k,
            home_advantage=home_advantage,
            mean_reversion_factor=mean_reversion_factor,
            initial_elo_promoted=initial_elo_promoted,
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_DIR / "laliga.csv")
    # study = run_elo_study(df, n_trials=100)
    # print("Best params:", study.best_params)
    # print("Best log loss:", study.best_value)
    # df_elo = calculate_elo(df, k=9.06, home_advantage=197,
    #                        mean_reversion_factor=0.0003, initial_elo_promoted=1234)
    # print("--- desviación del ELO por temporada ---")
    # print(df_elo.groupby("Season")["ELO_home"].std().round(1))
    # print(elo_val_log_loss(df, k=9.06, home_advantage=197,
    #                            mean_reversion_factor=0.0003,
    #                              initial_elo_promoted=1234))

    # print("--- mean_reversion_factor ---")
    # for f in [-0.1, 0, 0.1, 0.2, 0.3]:
    #     print(f, elo_val_log_loss(df, k=9.06, home_advantage=197,
    #                               mean_reversion_factor=f,
    #                               initial_elo_promoted=1234))

    # for p in [600, 1000, 1100, 1200, 1300, 1350, 1400, 1450, 1500]:
    #     print(p, elo_val_log_loss(df, k=9.06, home_advantage=197,
    #                             mean_reversion_factor=0.0003,
    #                             initial_elo_promoted=p))

    # for p in [600, 1000, 1100, 1200, 1300, 1350, 1400, 1450, 1500]:
    #     print(p, elo_val_log_loss(df, k=13.06, home_advantage=100,
    #                             mean_reversion_factor=0.0003,
    #                             initial_elo_promoted=p))
    # print("--- home_advantage ---")
    # for h in [40, 65, 100, 141]:
    #     print(h, elo_val_log_loss(df, k=8.3, home_advantage=h,
    #                               mean_reversion_factor=0.0,
    #                               initial_elo_promoted=1300))

    for f in [-0.1, -0.05, 0, 0.05, 0.1]:
        print(
            f,
            elo_val_log_loss(
                df,
                k=8,
                home_advantage=65,
                mean_reversion_factor=f,
                initial_elo_promoted=1300,
            ),
        )

    df_elo = calculate_elo(
        df,
        k=8,
        home_advantage=65,
        mean_reversion_factor=-0.1,
        initial_elo_promoted=1300,
    )
    print(df_elo.groupby("Season")["ELO_home"].agg(["std", "min", "max"]).round(0))
