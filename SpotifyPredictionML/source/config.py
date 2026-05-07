from pathlib import Path

# Root project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Raw Kaggle CSV files
HIGH_POPULARITY_FILE = RAW_DATA_DIR / "high_popularity_spotify_data.csv"
LOW_POPULARITY_FILE = RAW_DATA_DIR / "low_popularity_spotify_data.csv"

# Cleaned dataset
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "spotify_cleaned.csv"

# Output folders
MODELS_DIR = BASE_DIR / "models" / "genre_models"
REPORTS_DIR = BASE_DIR / "reports"

# Results file
RESULTS_FILE = REPORTS_DIR / "genre_model_results.csv"

# Number of genres to train models for
NUM_GENRES_TO_TRAIN = 6

# Minimum number of songs required to train a genre model
MIN_SONGS_PER_GENRE = 100

# Input features used by the model
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

# Target variable
TARGET_COLUMN = "track_popularity"

# Genre column
GENRE_COLUMN = "playlist_genre"