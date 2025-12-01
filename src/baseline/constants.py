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


F_USER_RATING_STD = "user_rating_std"
F_USER_MIN_RATING = "user_min_rating"
F_USER_MAX_RATING = "user_max_rating"
F_USER_RATING_RANGE = "user_rating_range"
F_USER_RELIABILITY = "user_reliability"
F_USER_BIAS = "user_bias"
F_USER_SHRUNK_MEAN = "user_shrunk_mean"
F_USER_RATING_TREND = "user_rating_trend"
F_USER_RECENCY_DAYS = "user_recency_days"
F_USER_RATING_FREQUENCY = "user_rating_frequency"

# Book features
F_BOOK_RATING_STD = "book_rating_std"
F_BOOK_MIN_RATING = "book_min_rating"
F_BOOK_MAX_RATING = "book_max_rating"
F_BOOK_RATING_RANGE = "book_rating_range"
F_BOOK_POPULARITY = "book_popularity"
F_BOOK_CONTROVERSIAL = "book_controversial"
F_BOOK_SHRUNK_MEAN = "book_shrunk_mean"
F_BOOK_QUALITY_SCORE = "book_quality_score"

# Author features
F_AUTHOR_RATINGS_COUNT = "author_ratings_count"
F_AUTHOR_RATING_STD = "author_rating_std"
F_AUTHOR_POPULARITY = "author_popularity"
F_AUTHOR_RELIABILITY = "author_reliability"

# Interaction features
F_HAS_READ_AUTHOR = "has_read_author"
F_USER_BOOK_RATING_DIFF = "user_book_rating_diff"
F_USER_BOOK_RATING_ABS_DIFF = "user_book_rating_abs_diff"
F_USER_BOOK_COMPATIBILITY = "user_book_compatibility"
F_USER_BOOK_POPULARITY_PRODUCT = "user_book_popularity_product"

# Combined features
F_COMBINED_RELIABILITY = "combined_reliability"
F_EXPECTED_RATING = "expected_rating"
F_PREDICTION_UNCERTAINTY = "prediction_uncertainty"
