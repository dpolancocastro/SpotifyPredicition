import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # -----------------------------
    # Load cleaned dataset
    # -----------------------------
    df = pd.read_csv("cleaned_spotify_data.csv")

    # -----------------------------
    # Define features and target
    # -----------------------------
    features = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "release_year"
    ]

    target = "popularity_class"

    # Ensure required columns exist
    required_columns = features + ["playlist_genre", target]
    df = df.dropna(subset=required_columns)

    # -----------------------------
    # Filter genres with enough data
    # -----------------------------
    MIN_SAMPLES = 100
    genre_counts = df["playlist_genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= MIN_SAMPLES].index

    df = df[df["playlist_genre"].isin(valid_genres)]

    print(f"Using {len(valid_genres)} genres with >= {MIN_SAMPLES} samples")

    # -----------------------------
    # Store results
    # -----------------------------
    results = []

    # -----------------------------
    # Loop through genres
    # -----------------------------
    for genre in valid_genres:
        print(f"\n{'=' * 60}")
        print(f"Training for genre: {genre}")
        print(f"{'=' * 60}")

        genre_df = df[df["playlist_genre"] == genre]

        X = genre_df[features]
        y = genre_df[target]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # -----------------------------
        # Baseline Model (Naive)
        # -----------------------------
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        y_base_pred = baseline.predict(X_test)
        baseline_acc = accuracy_score(y_test, y_base_pred)

        # -----------------------------
        # Real Model
        # -----------------------------
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_acc = accuracy_score(y_test, y_pred)

        # -----------------------------
        # Save results
        # -----------------------------
        results.append({
            "genre": genre,
            "num_samples": len(genre_df),
            "baseline_accuracy": baseline_acc,
            "model_accuracy": model_acc,
            "improvement": model_acc - baseline_acc
        })

        # -----------------------------
        # Print results
        # -----------------------------
        print(f"Samples: {len(genre_df)}")
        print(f"Baseline Accuracy: {baseline_acc:.3f}")
        print(f"Model Accuracy:    {model_acc:.3f}")
        print(f"Improvement:       {model_acc - baseline_acc:.3f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    # -----------------------------
    # Final Results Summary
    # -----------------------------
    results_df = pd.DataFrame(results)

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(results_df.sort_values(by="model_accuracy", ascending=False))

    # Save results
    results_df.to_csv("genre_model_results.csv", index=False)
    print("\nResults saved to genre_model_results.csv")


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    main()