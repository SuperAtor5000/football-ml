import pandas as pd
import numpy as np
from src.features.build_features import build_elo_and_rolling_features
from src.models.train import select_features


def season_from_date(date) -> str:
    """Return the season label (e.g. '2026/27') for a date."""
    date = pd.Timestamp(date)
    start_year = np.where(date.month >= 8, date.year, date.year - 1)
    end_year = (start_year + 1) % 100
    return f"{start_year}/{end_year:02d}"


def build_prediction_rows(history_df, matches, date=None):
    """Build the feature rows for future matches.

    history_df: played matches (same format as laliga.csv).
    matches: list of (home_team, away_team) tuples, one round, no team repeated.
    date: match date; defaults to tomorrow.
    Returns a DataFrame with the model's feature columns, one row per match.
    """
    date = (
        pd.Timestamp(date)
        if date is not None
        else pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    )
    date_str = date.strftime("%Y-%m-%d")  # same text format as the history
    season = season_from_date(date)

    known_teams = set(history_df["HomeTeam"]) | set(history_df["AwayTeam"])
    teams_in_round = [t for match in matches for t in match]
    unknown = [t for t in teams_in_round if t not in known_teams]
    if unknown:
        raise ValueError(f"Unknown teams: {unknown}")
    if len(teams_in_round) != len(set(teams_in_round)):
        raise ValueError("A team appears more than once in the round")

    future = pd.DataFrame(
        {
            "Date": date_str,
            "Season": season,
            "HomeTeam": [home for home, _ in matches],
            "AwayTeam": [away for _, away in matches],
        }
    )
    full_df = pd.concat([history_df, future], ignore_index=True)

    features_df = build_elo_and_rolling_features(full_df)
    is_future = (features_df["Date"] == date_str) & features_df["FTR"].isna()
    rows = features_df[is_future]

    return rows[select_features(include_market=False)].reset_index(drop=True)
