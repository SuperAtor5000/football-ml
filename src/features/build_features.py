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


def main():
    # Load the processed data
    processed_data_file = PROCESSED_DATA_DIR / "laliga.csv"
    df = pd.read_csv(processed_data_file)

    # Calculate ELO ratings
    df_with_elo = calculate_elo(df)

    # Save the DataFrame with ELO ratings to a new CSV file
    features_data_file = FEATURES_DATA_DIR / "laliga_features.csv"
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_with_elo.to_csv(features_data_file, index=False)
    print(f"Features data with ELO ratings saved to {features_data_file}")


if __name__ == "__main__":
    main()
