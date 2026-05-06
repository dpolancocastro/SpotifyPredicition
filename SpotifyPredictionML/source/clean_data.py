import pandas as pd

from config import (
    HIGH_POPULARITY_FILE,
    LOW_POPULARITY_FILE,
    CLEANED_DATA_FILE,
    PROCESSED_DATA_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    GENRE_COLUMN,
)


def load_raw_data():
    """
    Loads the high-popularity and low-popularity Spotify CSV files,
    then combines them into one dataframe.
    """

    high_df = pd.read_csv(HIGH_POPULARITY_FILE)
    low_df = pd.read_csv(LOW_POPULARITY_FILE)

    # Add a column so we still know which file each song came from
    high_df["popularity_group"] = "high"
    low_df["popularity_group"] = "low"

    combined_df = pd.concat([high_df, low_df], ignore_index=True)

    return combined_df


def clean_data(df):
    """
    Cleans the Spotify dataset for machine learning.
    """

    # Keep only the columns needed for this project
    columns_to_keep = FEATURE_COLUMNS + [TARGET_COLUMN, GENRE_COLUMN]
    df = df[columns_to_keep].copy()

    # Remove rows where the target or genre is missing
    df = df.dropna(subset=[TARGET_COLUMN, GENRE_COLUMN])

    # Standardize genre text
    df[GENRE_COLUMN] = df[GENRE_COLUMN].astype(str).str.strip().str.lower()

    # Convert features and target to numeric values
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing feature values with the median of each column
    for col in FEATURE_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    # Remove rows where the target could not be converted
    df = df.dropna(subset=[TARGET_COLUMN])

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df


def save_cleaned_data(df):
    """
    Saves the cleaned dataframe into data/processed/.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA_FILE, index=False)


def main():
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    save_cleaned_data(cleaned_df)

    print("Cleaning complete.")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Saved cleaned data to: {CLEANED_DATA_FILE}")


if __name__ == "__main__":
    main()