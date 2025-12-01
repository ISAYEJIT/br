"""
Project-wide constants.

This module defines constants that are part of the data schema or project
structure but are not intended to be tuned as hyperparameters.
"""

# --- FILENAMES ---
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
USER_DATA_FILENAME = "users.csv"
BOOK_DATA_FILENAME = "books.csv"
BOOK_GENRES_FILENAME = "book_genres.csv"
GENRES_FILENAME = "genres.csv"
BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
SUBMISSION_FILENAME = "submission.csv"
TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
NOMIC_EMBEDDINGS_FILENAME = "nomic_embeddings.pkl"
NOMIC_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
PROCESSED_DATA_FILENAME = "processed_features.parquet"

# --- COLUMN NAMES ---
# Main columns
COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_TARGET = "rating"
COL_SOURCE = "source"
COL_PREDICTION = "rating_predict"
COL_HAS_READ = "has_read"
COL_TIMESTAMP = "timestamp"

# Feature columns (newly created)
F_USER_MEAN_RATING = "user_mean_rating"
F_USER_RATINGS_COUNT = "user_ratings_count"
F_BOOK_MEAN_RATING = "book_mean_rating"
F_BOOK_RATINGS_COUNT = "book_ratings_count"
F_AUTHOR_MEAN_RATING = "author_mean_rating"
F_BOOK_GENRES_COUNT = "book_genres_count"

# Metadata columns from raw data
COL_GENDER = "gender"
COL_AGE = "age"
COL_AUTHOR_ID = "author_id"
COL_PUBLICATION_YEAR = "publication_year"
COL_LANGUAGE = "language"
COL_PUBLISHER = "publisher"
COL_AVG_RATING = "avg_rating"
COL_GENRE_ID = "genre_id"
COL_DESCRIPTION = "description"


# --- VALUES ---
VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"

# --- MAGIC NUMBERS ---
MISSING_CAT_VALUE = "-1"
MISSING_NUM_VALUE = -1
PREDICTION_MIN_VALUE = 0
PREDICTION_MAX_VALUE = 10


# Добавьте эти константы в соответствующие разделы:

# User features - ДОБАВИТЬ:
F_USER_SMOOTHED_MEAN = "user_smoothed_mean"
F_USER_BAYESIAN_MEAN = "user_bayesian_mean"
F_USER_RATING_PERCENTILE = "user_rating_percentile"
F_USER_POPULARITY_PERCENTILE = "user_popularity_percentile"
F_USER_RELIABILITY_ADVANCED = "user_reliability_advanced"
F_USER_STABILITY_SCORE = "user_stability_score"

# Book features - ДОБАВИТЬ:
F_BOOK_SMOOTHED_MEAN = "book_smoothed_mean"
F_BOOK_BAYESIAN_MEAN = "book_bayesian_mean"
F_BOOK_RATING_PERCENTILE = "book_rating_percentile"
F_BOOK_POPULARITY_PERCENTILE = "book_popularity_percentile"
F_BOOK_RELIABILITY_ADVANCED = "book_reliability_advanced"
F_BOOK_AGE = "book_age_at_reading"

F_AUTHOR_SMOOTHED_MEAN = "author_smoothed_mean"
F_AUTHOR_MEAN_RATING = "author_mean_rating"
F_AUTHOR_RATINGS_COUNT = "author_ratings_count"
F_AUTHOR_RATING_STD = "author_rating_std"
F_AUTHOR_POPULARITY = "author_popularity"
F_AUTHOR_RELIABILITY = "author_reliability"

# Temporal features - ДОБАВИТЬ:
F_DAYS_SINCE_LAST_READ = "days_since_last_read"
F_LOG_DAYS_SINCE_LAST = "log_days_since_last"
F_USER_RECENTLY_ACTIVE = "user_recently_active"

F_WEIGHTED_USER_BOOK_DIFF = "weighted_user_book_diff"
F_WEIGHTED_EXPECTED_RATING = "weighted_expected_rating"
F_SMOOTHED_USER_BOOK_DIFF = "smoothed_user_book_diff"
F_PERCENTILE_COMPATIBILITY = "percentile_compatibility"
F_PERCENTILE_SIMILARITY = "percentile_similarity"
F_POPULARITY_ALIGNMENT = "popularity_alignment"
F_ACTIVE_RELIABLE = "active_reliable"

F_EXPECTED_RATING_SMOOTHED = "expected_rating_smoothed"
F_FINAL_ENSEMBLE_RATING = "final_ensemble_rating"

F_USER_RATING_BIAS = "user_rating_bias"
F_BOOK_RATING_BIAS = "book_rating_bias"
F_RATING_STYLE_DIFFERENCE = "rating_style_difference"
F_RATING_STYLE_COMPATIBILITY = "rating_style_compatibility"
