from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

HIGH_POPULARITY_FILE = RAW_DATA_DIR / "high_popularity_spotify_data.csv"
LOW_POPULARITY_FILE = RAW_DATA_DIR / "low_popularity_spotify_data.csv"

CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "spotify_cleaned.csv"

MODELS_DIR = BASE_DIR / "models" / "genre_models"
REPORTS_DIR = BASE_DIR / "reports"


RESULTS_FILE = REPORTS_DIR / "genre_model_results.csv"

FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "tempo",
    "loudness",
    "liveness",
    "valence",
    "speechiness",
    "duration_ms",
    "acousticness",
    "instrumentalness",
    "mode",
    "key",
]

TARGET_COLUMN = "track_popularity"

GENRE_COLUMN = "playlist_genre"