import re

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    CLEANED_DATA_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    RESULTS_FILE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    GENRE_COLUMN,
    NUM_GENRES_TO_TRAIN,
    MIN_SONGS_PER_GENRE,
)


def make_safe_filename(text):
    """
    Converts a genre name into a safe filename.

    Example:
    "r&b" becomes "r_b"
    "latin pop" becomes "latin_pop"
    """

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")

    return text


def calculate_baseline(y_train, y_test):
    """
    Naive baseline:
    Predict the average popularity of the genre for every song.
    """

    genre_average = y_train.mean()

    baseline_predictions = np.full(
        shape=len(y_test),
        fill_value=genre_average
    )

    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))

    return baseline_mae, baseline_rmse


def choose_cv_folds(num_samples):
    """
    Chooses the number of cross-validation folds based on dataset size.

    Fewer than 100 samples:
        Skip the genre.

    100 to 249 samples:
        Use 3-fold cross-validation.

    250 or more samples:
        Use 5-fold cross-validation.
    """

    if num_samples < 100:
        return 0
    elif num_samples < 250:
        return 3
    else:
        return 5


def get_top_genres(df):
    """
    Selects the top genres with the most songs.

    This makes sure we train models for genres that have enough data.
    """

    genre_counts = df[GENRE_COLUMN].value_counts()

    eligible_genres = genre_counts[genre_counts >= MIN_SONGS_PER_GENRE]

    selected_genres = eligible_genres.head(NUM_GENRES_TO_TRAIN).index.tolist()

    print("\nSelected genres for training:")
    for genre in selected_genres:
        print(f"- {genre}: {genre_counts[genre]} songs")

    return selected_genres


def train_model_for_genre(genre_name, genre_df):
    """
    Trains and evaluates one Random Forest model for one genre.

    Evaluation process:
    1. Split genre data into 80% training and 20% testing.
    2. Calculate naive baseline on the test set.
    3. Run cross-validation on the training set.
    4. Train final model on all training data.
    5. Evaluate final model on the test set.
    6. Save the trained model.
    """

    num_songs = len(genre_df)

    if num_songs < MIN_SONGS_PER_GENRE:
        print(f"Skipping {genre_name}: not enough data ({num_songs} songs).")
        return None

    X = genre_df[FEATURE_COLUMNS]
    y = genre_df[TARGET_COLUMN]

    # Final holdout test set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Baseline evaluation on the final test set
    baseline_mae, baseline_rmse = calculate_baseline(y_train, y_test)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Choose cross-validation folds based on the size of the training set
    cv_folds = choose_cv_folds(len(X_train))

    cv_mae_mean = None
    cv_mae_std = None
    cv_rmse_mean = None
    cv_rmse_std = None
    cv_r2_mean = None
    cv_r2_std = None

    if cv_folds > 0:
        kfold = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42
        )

        # Scikit-learn returns negative MAE/RMSE because higher score is better.
        # We multiply by -1 later to convert them back to positive errors.
        cv_mae_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kfold,
            scoring="neg_mean_absolute_error"
        )

        cv_rmse_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kfold,
            scoring="neg_root_mean_squared_error"
        )

        cv_r2_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kfold,
            scoring="r2"
        )

        cv_mae_scores = -cv_mae_scores
        cv_rmse_scores = -cv_rmse_scores

        cv_mae_mean = cv_mae_scores.mean()
        cv_mae_std = cv_mae_scores.std()

        cv_rmse_mean = cv_rmse_scores.mean()
        cv_rmse_std = cv_rmse_scores.std()

        cv_r2_mean = cv_r2_scores.mean()
        cv_r2_std = cv_r2_scores.std()

    # Train final model on all training data
    model.fit(X_train, y_train)

    # Evaluate final model on the unseen test data
    predictions = model.predict(X_test)

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    model_r2 = r2_score(y_test, predictions)

    # Save trained model
    safe_genre_name = make_safe_filename(genre_name)
    model_path = MODELS_DIR / f"model_{safe_genre_name}.pkl"
    joblib.dump(model, model_path)

    result = {
        "genre": genre_name,
        "num_songs": num_songs,
        "cv_folds": cv_folds,

        "baseline_mae": baseline_mae,
        "model_mae": model_mae,

        "baseline_rmse": baseline_rmse,
        "model_rmse": model_rmse,

        "test_r2_score": model_r2,

        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std,

        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,

        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,

        "model_path": str(model_path),
    }

    return result


def train_selected_genre_models():
    """
    Loads the cleaned data and trains a separate model for the top selected genres.
    """

    df = pd.read_csv(CLEANED_DATA_FILE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    selected_genres = get_top_genres(df)

    if len(selected_genres) == 0:
        print("No genres have enough data to train a model.")
        return pd.DataFrame()

    results = []

    for genre in selected_genres:
        print(f"\nTraining model for genre: {genre}")

        genre_df = df[df[GENRE_COLUMN] == genre].copy()

        result = train_model_for_genre(genre, genre_df)

        if result is not None:
            results.append(result)

            print(f"Number of songs: {result['num_songs']}")
            print(f"CV folds: {result['cv_folds']}")

            print(f"Baseline MAE:  {result['baseline_mae']:.2f}")
            print(f"Model MAE:     {result['model_mae']:.2f}")

            print(f"Baseline RMSE: {result['baseline_rmse']:.2f}")
            print(f"Model RMSE:    {result['model_rmse']:.2f}")

            print(f"Final Test R² Score: {result['test_r2_score']:.2f}")

            if result["cv_mae_mean"] is not None:
                print(
                    f"CV MAE:  {result['cv_mae_mean']:.2f} "
                    f"± {result['cv_mae_std']:.2f}"
                )
                print(
                    f"CV RMSE: {result['cv_rmse_mean']:.2f} "
                    f"± {result['cv_rmse_std']:.2f}"
                )
                print(
                    f"CV R²:   {result['cv_r2_mean']:.2f} "
                    f"± {result['cv_r2_std']:.2f}"
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)

    print("\nTraining complete.")
    print(f"Results saved to: {RESULTS_FILE}")

    return results_df


if __name__ == "__main__":
    train_selected_genre_models()