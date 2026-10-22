import pandas as pd
import numpy as np
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DATA_DIR = PROJECT_ROOT / "data" / "features"

# ELO hyperparameters
K = 8
HOME_ADVANTAGE = 65
INITIAL_ELO = 1500
INITIAL_ELO_PROMOTED = 1300
MEAN_REVERSION_FACTOR = 0.0
ELO_SCALE = 400


def margin_of_victory_multiplier(
    home_goals: int, away_goals: int, math_function: str
) -> float:
    """Calculate the margin of victory multiplier based on the goal difference."""

    goal_difference = abs(home_goals - away_goals)

    if math_function == "none":
        return 1.0
    elif math_function == "linear":
        return float(max(goal_difference, 1))
    elif math_function == "logarithmic":
        if goal_difference == 0:
            return 1.0
        else:
            return float(1 + np.log(goal_difference))
    elif math_function == "square_root":
        if goal_difference == 0:
            return 1.0
        else:
            return float(np.sqrt(goal_difference))
    elif math_function == "world_football_elo":
        if goal_difference <= 1:
            return 1.0
        elif goal_difference == 2:
            return 1.5
        else:
            return float((11 + goal_difference) / 8)
    elif math_function == "harmonic":
        if goal_difference == 0:
            return 1.0
        else:
            return float(sum(1 / i for i in range(1, goal_difference + 1)))
    elif math_function == "harmonic_plus_one":
        return float(
            1 + sum(1 / i for i in range(1, goal_difference + 1))
        )  # Considered goal_difference=0 returns 1.0
    elif math_function == "saturated":
        if goal_difference == 0:
            return 1.0
        else:
            return float(1 + (goal_difference - 1) / (goal_difference))
    elif math_function == "capped":
        return float(max(min(goal_difference, 3), 1))
    else:
        raise ValueError(f"Unknown math_function: {math_function}")


def compute_margin_scale_factor(df_matches: pd.DataFrame, math_function: str) -> float:
    """Mean multiplier over the given matches (used to keep the average k unchanged)."""
    home_goals = df_matches["FTHG"]
    away_goals = df_matches["FTAG"]
    return float(
        np.mean(
            [
                margin_of_victory_multiplier(h, a, math_function)
                for h, a in zip(home_goals, away_goals)
            ]
        )
    )


def calculate_elo(
    df: pd.DataFrame,
    k=K,
    home_advantage=HOME_ADVANTAGE,
    initial_elo=INITIAL_ELO,
    initial_elo_promoted=INITIAL_ELO_PROMOTED,
    mean_reversion_factor=MEAN_REVERSION_FACTOR,
    elo_scale=ELO_SCALE,
    margin_function="none",
    margin_scale=1.0,
) -> pd.DataFrame:
    """Calculate ELO ratings for each team updated match by match."""

    # Initialize ELO ratings for all teams
    # In the first season, all teams start with INITIAL_ELO, we dont take into account the previous season
    # All promoted teams start with INITIAL_ELO_PROMOTED as well, we dont take into account their previous season in the lower division

    df = df.sort_values("Date").reset_index(drop=True)
    elo_ratings = {}
    home_elos_list = []
    away_elos_list = []
    current_season = None
    previous_season_teams = set()
    current_season_teams = set()
    initialized_this_season = set()

    for idx, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        season = row["Season"]

        if season != current_season:
            previous_season_teams = current_season_teams.copy()
            current_season_teams = set()
            initialized_this_season = set()

            # Apply mean reversion to all teams when season changes
            if current_season is not None:
                for team in elo_ratings:
                    elo_ratings[team] = (
                        elo_ratings[team] * (1 - mean_reversion_factor)
                        + initial_elo * mean_reversion_factor
                    )

            current_season = season

        current_season_teams.add(home_team)
        current_season_teams.add(away_team)

        # Reset ELO once per team per season
        for team in [home_team, away_team]:
            if team not in initialized_this_season:
                initialized_this_season.add(team)
                if not previous_season_teams:
                    elo_ratings[team] = initial_elo
                elif team not in previous_season_teams:
                    elo_ratings[team] = initial_elo_promoted

        # Save the current ELO ratings before the match
        elo_home_rating = elo_ratings[home_team]
        elo_away_rating = elo_ratings[away_team]

        # Calculate expected scores based on ELO ratings and home advantage
        expected_home_score_win = 1 / (
            1 + 10 ** ((elo_away_rating - elo_home_rating - home_advantage) / elo_scale)
        )
        expected_away_score_win = 1 - expected_home_score_win

        # Save the ELO ratings for the current match
        home_elos_list.append(elo_home_rating)
        away_elos_list.append(elo_away_rating)

        # Determine actual scores based on match result
        if row["FTR"] == "H":
            actual_home_score = 1.0
        elif row["FTR"] == "D":
            actual_home_score = 0.5
        else:
            actual_home_score = 0.0
        actual_away_score = 1 - actual_home_score

        # Update ELO ratings based on the match result
        g = (
            margin_of_victory_multiplier(row["FTHG"], row["FTAG"], margin_function)
            / margin_scale
        )
        elo_ratings[home_team] += k * g * (actual_home_score - expected_home_score_win)
        elo_ratings[away_team] += k * g * (actual_away_score - expected_away_score_win)

    df["ELO_home"] = home_elos_list
    df["ELO_away"] = away_elos_list
    df["ELO_diff"] = df["ELO_home"] - df["ELO_away"]

    return df


def build_team_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame with team-level features."""

    home_rows = []
    away_rows = []

    for _, row in df.iterrows():
        # Home team row
        home_rows.append(
            {
                "Date": row["Date"],
                "Season": row["Season"],
                "team": row["HomeTeam"],
                "opponent": row["AwayTeam"],
                "is_home": True,
                "goals_scored": row["FTHG"],
                "goals_conceded": row["FTAG"],
                "shots_on_target": row["HST"],
                "shots_on_target_conceded": row["AST"],
                "corners": row["HC"],
                "corners_conceded": row["AC"],
                "fouls": row["HF"],
                "fouls_conceded": row["AF"],
                "points": 3 if row["FTR"] == "H" else 1 if row["FTR"] == "D" else 0,
            }
        )

        # Away team row
        away_rows.append(
            {
                "Date": row["Date"],
                "Season": row["Season"],
                "team": row["AwayTeam"],
                "opponent": row["HomeTeam"],
                "is_home": False,
                "goals_scored": row["FTAG"],
                "goals_conceded": row["FTHG"],
                "shots_on_target": row["AST"],
                "shots_on_target_conceded": row["HST"],
                "corners": row["AC"],
                "corners_conceded": row["HC"],
                "fouls": row["AF"],
                "fouls_conceded": row["HF"],
                "points": 3 if row["FTR"] == "A" else 1 if row["FTR"] == "D" else 0,
            }
        )

    team_df = pd.concat([pd.DataFrame(home_rows), pd.DataFrame(away_rows)])
    team_df = team_df.sort_values(["team", "Date"]).reset_index(drop=True)
    team_df["matchday"] = team_df.groupby(["team", "Season"]).cumcount() + 1

    return team_df


def add_standings_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match standings features (points, distance to Europe/relegation)."""
    team_df["points_accumulated_before_match"] = (
        team_df.groupby(["team", "Season"])["points"].cumsum() - team_df["points"]
    )
    team_df["points_remaining"] = (
        38 - (team_df["matchday"] - 1)
    ) * 3  # Assuming 38 matches in a season

    thresholds = compute_threshold_points(team_df)
    team_df = team_df.merge(thresholds, on=["Date", "Season"], how="left")
    team_df["dist_europe"] = (
        team_df["points_accumulated_before_match"] - team_df["points_7th"]
    )
    team_df["dist_relegation"] = (
        team_df["points_accumulated_before_match"] - team_df["points_18th"]
    )

    return team_df.drop(columns=["points_7th", "points_18th"])


def compute_threshold_points(team_df: pd.DataFrame) -> pd.DataFrame:
    """Points of the 7th and 18th team before each date, per season."""
    df = team_df.copy()
    df["points_cum"] = df.groupby(["team", "Season"])["points"].cumsum()

    rows = []
    for season, df_season in df.groupby("Season"):
        table = df_season.pivot(index="Date", columns="team", values="points_cum")
        table = table.ffill().fillna(0)  # teams that haven't played yet have 0 points
        table = table.shift(1)
        table = table.fillna(0)
        sorted_points = -np.sort(-table.values, axis=1)  # each row sorted high -> low
        out = pd.DataFrame(
            {
                "Date": table.index,
                "Season": season,
                "points_7th": sorted_points[:, 6],
                "points_18th": sorted_points[:, 17],
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def calculate_rolling_features(
    team_df: pd.DataFrame, windows_matches: list[int] = [3, 5, 7, 10]
) -> pd.DataFrame:
    """Calculate rolling features for each team based on specified windows."""

    stats = [
        "goals_scored",
        "goals_conceded",
        "shots_on_target",
        "shots_on_target_conceded",
        "corners",
        "corners_conceded",
        "fouls",
        "fouls_conceded",
        "points",
    ]

    for window in windows_matches:
        for stat in stats:
            team_df[f"{stat}_last{window}_matches"] = team_df.groupby(
                ["team", "Season"]
            )[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    return team_df


def merge_rolling_features(
    df_with_elo: pd.DataFrame, team_df_with_rolling_features: pd.DataFrame
) -> pd.DataFrame:
    """Merge rolling features back to the original DataFrame."""

    standings_cols = [
        "points_accumulated_before_match",
        "points_remaining",
        "dist_europe",
        "dist_relegation",
    ]

    # Rename rolling feature columns to distinguish between home and away teams
    rolling_cols = [
        c
        for c in team_df_with_rolling_features.columns
        if "_last" in c or c == "matchday" or c in standings_cols
    ]

    # Version for home team
    home_df = team_df_with_rolling_features[
        ["Date", "Season", "team"] + rolling_cols
    ].rename(columns={col: f"HomeTeam_{col}" for col in rolling_cols})

    # Version for away team
    away_df = team_df_with_rolling_features[
        ["Date", "Season", "team"] + rolling_cols
    ].rename(columns={col: f"AwayTeam_{col}" for col in rolling_cols})

    # Merge
    final_df = df_with_elo.merge(
        home_df,
        left_on=["Date", "Season", "HomeTeam"],
        right_on=["Date", "Season", "team"],
        how="left",
    ).drop(columns="team")

    final_df = final_df.merge(
        away_df,
        left_on=["Date", "Season", "AwayTeam"],
        right_on=["Date", "Season", "team"],
        how="left",
    ).drop(columns="team")

    return final_df


def build_elo_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ELO and rolling features for the dataset."""

    # Calculate ELO ratings
    df_with_elo = calculate_elo(df)

    # Build team-level DataFrame
    team_df = build_team_dataframe(df_with_elo)

    # Calculate rolling features
    team_df_with_rolling_features = calculate_rolling_features(team_df)

    # Add standings features (points before match, distance to Europe/relegation)
    team_df_with_rolling_and_standings_features = add_standings_features(
        team_df_with_rolling_features
    )

    # Merge the rolling features back to the original DataFrame
    final_df = merge_rolling_features(
        df_with_elo, team_df_with_rolling_and_standings_features
    )

    return final_df


def main():
    # Load the processed data
    processed_data_file = PROCESSED_DATA_DIR / "laliga.csv"
    df = pd.read_csv(processed_data_file)

    # Calculate ELO ratings
    df_with_elo_and_rolling_features = build_elo_and_rolling_features(df)

    # Save the DataFrame with ELO ratings to a new CSV file
    features_data_file = FEATURES_DATA_DIR / "laliga_features.csv"
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_with_elo_and_rolling_features.to_csv(features_data_file, index=False)
    print(f"Features data with ELO ratings saved to {features_data_file}")


if __name__ == "__main__":
    main()
