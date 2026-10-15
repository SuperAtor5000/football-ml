import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.models.train import BEST_XGB_PARAMS, df, select_features

# Each fold: (training seasons, validation season)
SEASONS = [
    "2015/16",
    "2016/17",
    "2017/18",
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
]
FOLDS = [(SEASONS[:i], SEASONS[i]) for i in range(3, len(SEASONS))]


def walk_forward_losses(df, feature_columns, params, folds):
    """Return a list with one array of per-match losses for each fold."""
    losses_per_fold = []
    for train_seasons, val_season in folds:
        df_train = df[df["Season"].isin(train_seasons)]
        df_val = df[df["Season"] == val_season]

        X_train = df_train[feature_columns]
        X_val = df_val[feature_columns]

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(df_train["FTR"])
        y_val = encoder.transform(df_val["FTR"])

        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)

        # loss of each match = -log(probability given to the real result)
        losses = -np.log(proba[np.arange(len(y_val)), y_val])
        losses_per_fold.append(losses)
    return losses_per_fold


def paired_bootstrap(losses_a, losses_b, n_boot=2000, seed=42):
    """Return the mean loss difference (A - B) and its 95% bootstrap interval.

    Negative mean: A is better than B (lower loss).
    """
    array_losses_a = np.concatenate(losses_a)
    array_losses_b = np.concatenate(losses_b)
    if len(array_losses_a) != len(array_losses_b):
        raise ValueError("Arrays must have the same length")

    diff = array_losses_a - array_losses_b
    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        indices = rng.integers(0, len(diff), size=len(diff))
        boot_means[i] = diff[indices].mean()

    diff_mean = diff.mean()
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return diff_mean, (ci_low, ci_high)


if __name__ == "__main__":
    feature_columns_a = select_features(include_market=True)
    feature_columns_b = select_features(include_market=False)
    losses_a_per_fold = walk_forward_losses(
        df, feature_columns_a, BEST_XGB_PARAMS, FOLDS
    )
    losses_b_per_fold = walk_forward_losses(
        df, feature_columns_b, BEST_XGB_PARAMS, FOLDS
    )
    for (_, val_season), losses_a, losses_b in zip(
        FOLDS, losses_a_per_fold, losses_b_per_fold
    ):
        print(
            val_season,
            f"A={losses_a.mean():.4f}",
            f"B={losses_b.mean():.4f}",
            f"diff={losses_a.mean() - losses_b.mean():+.4f}",
        )
    diff_mean, ci = paired_bootstrap(losses_a_per_fold, losses_b_per_fold)
    print(f"Paired bootstrap: {diff_mean:.4f} ({ci[0]:.4f}, {ci[1]:.4f})")
