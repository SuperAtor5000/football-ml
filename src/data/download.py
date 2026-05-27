import pathlib
import requests
from datetime import date

BASE_URL = "https://www.football-data.co.uk/mmz4281"
COMPETITION = "SP1"
RAW_DATA_DIR = pathlib.Path("data/raw/laliga")

# First season start year (2013 → season 2013/14)
FIRST_SEASON_START_YEAR = 2013


def season_code(start_year: int) -> str:
    """Modifies a start year into a season code (e.g., 2013 → '1314')."""
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def csv_url(start_year: int) -> str:
    """Builds the download URL for a given season start year."""
    return f"{BASE_URL}/{season_code(start_year)}/{COMPETITION}.csv"


def current_season_start() -> int:
    """Actual season start year based on the current date. If it's August or later, the season has already started."""
    today = date.today()
    year = today.year
    if today.month >= 8:  # If it's August or later, the season has already started
        return year
    return year - 1


def all_seasons() -> list[int]:
    """List of start years for all seasons from FIRST_SEASON_START_YEAR to the current one."""
    return list(range(FIRST_SEASON_START_YEAR, current_season_start() + 1))


def download_season_data(start_year: int) -> None:
    url = csv_url(start_year)
    code = season_code(start_year)
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the download fails
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_DIR / f"SP1_{code}.csv"
    with open(file_path, "wb") as f:
        f.write(response.content)
    print(f"Data for season {code} downloaded successfully.")


def download_all_seasons() -> None:
    for start_year in all_seasons():
        download_season_data(start_year)


if __name__ == "__main__":
    download_all_seasons()
