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


def compare_feature_sets(
    df, feature_columns_a, feature_columns_b, params, folds, name_a="A", name_b="B"
):
    """Compare two feature sets with walk-forward validation and a paired bootstrap.

    diff = loss(A) - loss(B). Negative: A is better. Positive: B is better.
    """
    if len(feature_columns_a) != len(set(feature_columns_a)):
        raise ValueError("Duplicate feature columns")
    if len(feature_columns_b) != len(set(feature_columns_b)):
        raise ValueError("Duplicate feature columns")

    losses_a_per_fold = walk_forward_losses(df, feature_columns_a, params, folds)
    losses_b_per_fold = walk_forward_losses(df, feature_columns_b, params, folds)

    wins_b = 0
    for (_, val_season), losses_a, losses_b in zip(
        folds, losses_a_per_fold, losses_b_per_fold
    ):
        diff = (
            losses_a.mean() - losses_b.mean()
        )  # mean of A minus mean of B for this season
        if diff > 0:  # when does B win this fold?
            wins_b += 1
        print(
            val_season,
            f"{name_a}={losses_a.mean():.4f}",
            f"{name_b}={losses_b.mean():.4f}",
            f"diff={diff:+.4f}",
        )

    # 3. Bootstrap on all matches together
    diff_mean, ci = paired_bootstrap(losses_a_per_fold, losses_b_per_fold)
    print(f"Paired bootstrap: {diff_mean:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f})")
    print(f"{name_b} better in {wins_b}/{len(folds)} folds")
    adopt = wins_b >= 4 and ci[0] > 0
    print(f"Verdict: {'ADOPT ' + name_b if adopt else 'do NOT adopt ' + name_b}")

    return diff_mean, ci, wins_b


if __name__ == "__main__":
    base = select_features(include_market=False)
    compare_feature_sets(
        df,
        base,
        base + ["crowd_fraction"],
        BEST_XGB_PARAMS,
        FOLDS,
        name_a="base",
        name_b="candidate",
    )
