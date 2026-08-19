import logging
import pathlib
import pandas as pd
import numpy as np

# Get project root (one level up from notebooks/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "laliga"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "laliga.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load all the CSVs from dir raw/ and concatenate them into a single DataFrame."""
    all_files = sorted(RAW_DATA_DIR.glob("SP1_*.csv"))
    df_list = []
    for file in all_files:
        log.info(f"Loading {file.name}...")
        df_list.append(pd.read_csv(file))
    df = pd.concat(df_list, ignore_index=True)
    log.info(f"Combined data: {len(df)} rows.")
    log.info(f"Loaded {len(all_files)} files, {len(df)} matches in total.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the raw DataFrame."""

    log.info("Cleaning data...")

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    log.info(f"Data cleaned. {df['Date'].isnull().sum()} rows with invalid dates.")

    # Add new column: TotalGoals
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]

    # Add new column: Season (e.g., season starting in 2013 and finishing in 2014 to season 2013/14)
    year = df["Date"].dt.year
    month = df["Date"].dt.month
    season_start = np.where(month >= 8, year, year - 1)
    season_end = (season_start + 1) % 100
    df["Season"] = (
        pd.Series(season_start, index=df.index).astype(str)
        + "/"
        + pd.Series(season_end, index=df.index).astype(str).str.zfill(2)
    )
    log.info("Added TotalGoals and Season columns.")
    return df


def normalize_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize bookmaker odds to remove margin and get true probabilities."""

    log.info("Normalizing bookmaker odds...")
    bookmakers = {
        "B365": ("B365H", "B365D", "B365A"),
        "BW": ("BWH", "BWD", "BWA"),
        "PS": ("PSH", "PSD", "PSA"),
        "WH": ("WHH", "WHD", "WHA"),
    }

    for bk, (h_col, d_col, a_col) in bookmakers.items():
        df[f"{bk}_implied_H"] = 1 / df[h_col]
        df[f"{bk}_implied_D"] = 1 / df[d_col]
        df[f"{bk}_implied_A"] = 1 / df[a_col]
        df[f"{bk}_sum"] = (
            df[f"{bk}_implied_H"] + df[f"{bk}_implied_D"] + df[f"{bk}_implied_A"]
        )
        df[f"{bk}_norm_H"] = df[f"{bk}_implied_H"] / df[f"{bk}_sum"]
        df[f"{bk}_norm_D"] = df[f"{bk}_implied_D"] / df[f"{bk}_sum"]
        df[f"{bk}_norm_A"] = df[f"{bk}_implied_A"] / df[f"{bk}_sum"]

    bookmakers_closing = {
        "B365": ("B365CH", "B365CD", "B365CA"),
        "BW": ("BWCH", "BWCD", "BWCA"),
        "PS": ("PSCH", "PSCD", "PSCA"),
        "WH": ("WHCH", "WHCD", "WHCA"),
    }

    for bk, (h_col, d_col, a_col) in bookmakers_closing.items():
        df[f"{bk}_implied_CH"] = 1 / df[h_col]
        df[f"{bk}_implied_CD"] = 1 / df[d_col]
        df[f"{bk}_implied_CA"] = 1 / df[a_col]
        df[f"{bk}_sum_C"] = (
            df[f"{bk}_implied_CH"] + df[f"{bk}_implied_CD"] + df[f"{bk}_implied_CA"]
        )
        df[f"{bk}_norm_CH"] = df[f"{bk}_implied_CH"] / df[f"{bk}_sum_C"]
        df[f"{bk}_norm_CD"] = df[f"{bk}_implied_CD"] / df[f"{bk}_sum_C"]
        df[f"{bk}_norm_CA"] = df[f"{bk}_implied_CA"] / df[f"{bk}_sum_C"]

    # Drop intermediate columns
    cols_to_drop = [c for c in df.columns if "_implied_" in c or "_sum" in c]
    df = df.drop(columns=cols_to_drop)

    log.info(
        f"Normalized opening odds for {len(bookmakers)} bookmakers and closing odds for {len(bookmakers_closing)} bookmakers."
    )

    return df


def add_market_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Add market movement features based on opening/closing odds."""
    movements_H = [
        df["PS_norm_CH"] - df["PS_norm_H"],
        df["B365_norm_CH"] - df["B365_norm_H"],
        df["BW_norm_CH"] - df["BW_norm_H"],
        df["WH_norm_CH"] - df["WH_norm_H"],
    ]
    movements_D = [
        df["PS_norm_CD"] - df["PS_norm_D"],
        df["B365_norm_CD"] - df["B365_norm_D"],
        df["BW_norm_CD"] - df["BW_norm_D"],
        df["WH_norm_CD"] - df["WH_norm_D"],
    ]
    movements_A = [
        df["PS_norm_CA"] - df["PS_norm_A"],
        df["B365_norm_CA"] - df["B365_norm_A"],
        df["BW_norm_CA"] - df["BW_norm_A"],
        df["WH_norm_CA"] - df["WH_norm_A"],
    ]

    df["market_movement_H"] = pd.concat(movements_H, axis=1).mean(axis=1)
    df["market_movement_D"] = pd.concat(movements_D, axis=1).mean(axis=1)
    df["market_movement_A"] = pd.concat(movements_A, axis=1).mean(axis=1)
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select the desired columns for the final dataset."""

    log.info("Selecting relevant columns for the final dataset")

    selected_columns = [
        "Date",
        "Season",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HTHG",
        "HTAG",
        "HTR",
        "TotalGoals",
        "HS",
        "AS",
        "HST",
        "AST",
        "HF",
        "AF",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
        "B365_norm_H",
        "B365_norm_D",
        "B365_norm_A",
        "BW_norm_H",
        "BW_norm_D",
        "BW_norm_A",
        "PS_norm_H",
        "PS_norm_D",
        "PS_norm_A",
        "WH_norm_H",
        "WH_norm_D",
        "WH_norm_A",
        "B365_norm_CH",
        "B365_norm_CD",
        "B365_norm_CA",
        "BW_norm_CH",
        "BW_norm_CD",
        "BW_norm_CA",
        "PS_norm_CH",
        "PS_norm_CD",
        "PS_norm_CA",
        "WH_norm_CH",
        "WH_norm_CD",
        "WH_norm_CA",
        "market_movement_H",
        "market_movement_D",
        "market_movement_A",
    ]

    log.info(f"Selected columns for the final dataset: {len(selected_columns)}")

    return df[selected_columns]


def save_processed_data(df: pd.DataFrame) -> None:
    """Save the processed DataFrame to a CSV file."""

    log.info(f"Saving processed data to {OUTPUT_FILE}...")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    log.info(f"Processed data saved to {OUTPUT_FILE}.")


def main():
    """Main function to load, clean, normalize, select columns, and save the processed data."""

    df_raw = load_raw_data()
    df_cleaned = clean_data(df_raw)
    df_normalized = normalize_odds(df_cleaned)
    df_market_movement = add_market_movement(df_normalized)
    df_final = select_columns(df_market_movement)
    save_processed_data(df_final)
    log.info("Data processing completed successfully.")


if __name__ == "__main__":
    main()
