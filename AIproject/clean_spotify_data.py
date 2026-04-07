import argparse
from pathlib import Path
import pandas as pd

# Expected columns are based on the Kaggle dataset:
# https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset
# Common columns reported for the dataset include:
# energy, tempo, danceability, playlist_genre, loudness, liveness, valence,
# track_artist, time_signature, speechiness, track_popularity, track_href,
# uri, track_album, playlist_name, analysis_url, track_id, track_name,
# track_album_release_date, instrumentalness, track_album_id, mode, key,
# duration_ms, acousticness, id, playlist_subgenre, type, playlist_id

DROP_COLUMNS_IF_PRESENT = [
    "track_href",
    "uri",
    "analysis_url",
    "track_id",
    "track_album_id",
    "id",
    "type",
    "playlist_id",
    "track_name",
    "track_artist",
    "track_album",
    "playlist_name",
]

NUMERIC_COLUMNS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
    "track_popularity",
]

CATEGORICAL_COLUMNS = [
    "playlist_genre",
    "playlist_subgenre",
    "track_album_release_date",
]

FEATURE_COLUMNS_FOR_MODELING = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
    "playlist_genre",
    "playlist_subgenre",
    "release_year",
]


def popularity_to_class(value: float) -> str:
    if value < 40:
        return "low"
    if value < 70:
        return "medium"
    return "high"



def parse_release_year(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return pd.NA



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names lightly
    df.columns = [col.strip() for col in df.columns]

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Drop metadata columns (not needed)
    cols_to_drop = [c for c in DROP_COLUMNS_IF_PRESENT if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Convert expected numeric columns to numeric in case some were read as objects
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse release year from the release date column
    if "track_album_release_date" in df.columns:
        df["release_year"] = df["track_album_release_date"].apply(parse_release_year)

    # Drop rows missing the target
    if "track_popularity" not in df.columns:
        raise ValueError(
            "Expected column 'track_popularity' was not found. "
            "Check that you are using the Spotify Music Dataset from Kaggle."
        )
    df = df.dropna(subset=["track_popularity"])

    # Fill missing numeric values with median
    numeric_cols_present = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols_present:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Fill missing categorical values with mode if available, otherwise 'unknown'
    categorical_cols_present = df.select_dtypes(exclude="number").columns.tolist()
    for col in categorical_cols_present:
        if df[col].mode(dropna=True).empty:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    # Remove impossible popularity values if any bad rows slipped in
    df = df[(df["track_popularity"] >= 0) & (df["track_popularity"] <= 100)]

    # Create a categorical target for classification
    df["popularity_class"] = df["track_popularity"].apply(popularity_to_class)

    keep_cols = [c for c in FEATURE_COLUMNS_FOR_MODELING if c in df.columns]
    keep_cols += ["track_popularity", "popularity_class"]
    df = df[keep_cols]

    return df



def read_input_files(input_paths):
    frames = []
    for path in input_paths:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        frames.append(pd.read_csv(file_path))
    if not frames:
        raise ValueError("No input files were provided.")
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]



def main():
    parser = argparse.ArgumentParser(
        description="Clean the Kaggle Spotify Music Dataset into a modeling-ready CSV."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more input CSV files. You can pass one merged file, or two files "
            "such as high_popularity_spotify_data.csv and low_popularity_spotify_data.csv."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="cleaned_spotify_data.csv",
        help="Output CSV filename. Default: cleaned_spotify_data.csv",
    )
    args = parser.parse_args()

    raw_df = read_input_files(args.inputs)
    cleaned_df = clean_dataframe(raw_df)
    cleaned_df.to_csv(args.output, index=False)

    print("Cleaning complete.")
    print(f"Rows: {len(cleaned_df):,}")
    print(f"Columns: {len(cleaned_df.columns):,}")
    print(f"Saved cleaned file to: {args.output}")
    print("Columns kept:")
    for col in cleaned_df.columns:
        print(f"- {col}")


if __name__ == "__main__":
    main()
