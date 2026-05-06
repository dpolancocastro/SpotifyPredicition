# Spotify Popularity Prediction by Genre

## Project Overview

This project uses a Spotify music dataset to predict song popularity based on audio features such as danceability, energy, tempo, loudness, valence, acousticness, instrumentalness, and more.

Instead of training one model for all songs, this project trains a separate machine learning model for each playlist genre.

The main goal is to test whether genre-specific machine learning models can predict track popularity better than a simple baseline.

## Dataset

The dataset comes from Kaggle's Spotify Music Dataset by Solomon Ameh.

The project expects two CSV files:

high_popularity_spotify_data.csv
low_popularity_spotify_data.csv


These files should be placed inside:


data/raw/


The raw data files are not included in this repository because datasets can be large and may have usage restrictions.

## Target Variable

The target variable is:


track_popularity


This is the value the model tries to predict.

## Input Features

The model uses the following input features:

```text
danceability
energy
tempo
loudness
liveness
valence
speechiness
duration_ms
acousticness
instrumentalness
mode
key
```

## Genre-Specific Modeling

The dataset is divided by playlist genre.

For each genre, the project trains a separate Random Forest regression model.

For example:

```text
One model for pop
One model for rap
One model for rock
One model for edm
```

This approach was chosen because different genres may have different relationships between audio features and popularity.

## Baseline

The naive baseline predicts the average popularity of songs within each genre.

For example, if the average popularity of pop songs in the training set is 65, then the baseline predicts 65 for every pop song in the test set.

The machine learning model is considered useful if it performs better than this baseline.

## Model

The project uses a Random Forest Regressor.

A Random Forest is an ensemble machine learning model that builds many decision trees and averages their predictions.

## Evaluation

The project uses:

```text
MAE
RMSE
R² score
Cross-validation
```

### MAE

Mean Absolute Error measures the average prediction mistake.

Lower MAE is better.

### RMSE

Root Mean Squared Error is similar to MAE, but it penalizes larger errors more strongly.

Lower RMSE is better.

### R² Score

R² measures how much variation in popularity the model explains.

A score closer to 1 is better.

## Train/Test Split

For each genre, the data is split into:

```text
80% training data
20% testing data
```

The training data is used to train the model.

The testing data is used to evaluate how well the model performs on songs it has not seen before.

## Cross-Validation

Cross-validation is performed on the training data.

The number of folds depends on the size of the genre dataset:

```text
Fewer than 100 songs: skipped
100 to 249 songs: 3-fold cross-validation
250 or more songs: 5-fold cross-validation
```

This helps evaluate how stable the model is across different training subsets.

## Project Structure

```text
spotify_ai_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── genre_models/
│
├── reports/
│
├── src/
│   ├── config.py
│   ├── clean_data.py
│   ├── train_by_genre.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project

### 1. Create a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On Mac/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place these files inside `data/raw/`:

```text
high_popularity_spotify_data.csv
low_popularity_spotify_data.csv
```

### 4. Clean the data

```bash
python src/clean_data.py
```

This creates:

```text
data/processed/spotify_cleaned.csv
```

### 5. Train the models

```bash
python src/train_by_genre.py
```

This creates:

```text
models/genre_models/
reports/genre_model_results.csv
```

### 6. Run the full pipeline

```bash
python src/main.py
```

## Output

The trained genre-specific models are saved in:

```text
models/genre_models/
```

The model results are saved in:

```text
reports/genre_model_results.csv
```

## How to Interpret Results

A model is considered successful if:

```text
model_mae < baseline_mae
model_rmse < baseline_rmse
test_r2_score > 0
```

For example, if the baseline MAE is 12.5 and the model MAE is 8.9, then the machine learning model performed better than simply predicting the genre average.

## Summary

This project demonstrates a supervised machine learning approach to predicting Spotify song popularity. By training separate models for each genre and comparing them to a genre-average baseline, the project tests whether audio features can provide useful predictive information about popularity.