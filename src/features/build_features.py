import pandas as pd
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DATA_DIR = PROJECT_ROOT / "data" / "features"

# ELO hyperparameters
K = 20
HOME_ADVANTAGE = 65
INITIAL_ELO = 1500
INITIAL_ELO_PROMOTED = 1450
MEAN_REVERSION_FACTOR = 0.2
ELO_SCALE = 400


def calculate_elo(df: pd.DataFrame) -> pd.DataFrame:
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
                        elo_ratings[team] * (1 - MEAN_REVERSION_FACTOR)
                        + INITIAL_ELO * MEAN_REVERSION_FACTOR
                    )

            current_season = season

        current_season_teams.add(home_team)
        current_season_teams.add(away_team)

        # Reset ELO once per team per season
        for team in [home_team, away_team]:
            if team not in initialized_this_season:
                initialized_this_season.add(team)
                if not previous_season_teams:
                    elo_ratings[team] = INITIAL_ELO
                elif team not in previous_season_teams:
                    elo_ratings[team] = INITIAL_ELO_PROMOTED

        # Save the current ELO ratings before the match
        elo_home_rating = elo_ratings[home_team]
        elo_away_rating = elo_ratings[away_team]

        # Calculate expected scores based on ELO ratings and home advantage
        expected_home_score_win = 1 / (
            1 + 10 ** ((elo_away_rating - elo_home_rating - HOME_ADVANTAGE) / ELO_SCALE)
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
        elo_ratings[home_team] += K * (actual_home_score - expected_home_score_win)
        elo_ratings[away_team] += K * (actual_away_score - expected_away_score_win)

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

    # Columnas rolling que quieres renombrar
    rolling_cols = [
        c
        for c in team_df_with_rolling_features.columns
        if "_last" in c or c == "matchday"
    ]

    # Versión para equipo local
    home_df = team_df_with_rolling_features[
        ["Date", "Season", "team"] + rolling_cols
    ].rename(columns={col: f"HomeTeam_{col}" for col in rolling_cols})

    # Versión para equipo visitante
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

    # Merge the rolling features back to the original DataFrame
    final_df = merge_rolling_features(df_with_elo, team_df_with_rolling_features)

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
