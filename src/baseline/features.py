"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from pathlib import Path

from . import config, constants


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Безопасные агрегатные фичи без leakage.
    Использует только существующие константы.
    """
    print("Adding safe aggregate features...")

    train_read = train_df[train_df[constants.COL_HAS_READ] == 1].copy()

    global_mean = train_read[config.TARGET].mean()
    global_std = train_read[config.TARGET].std()

    user_agg = train_read.groupby(constants.COL_USER_ID)[config.TARGET].agg(['mean', 'count']).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT
    ]

    book_agg = train_read.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(['mean', 'count']).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT
    ]

    author_agg = None
    books_df = None

    possible_paths = [
        config.DATA_DIR / "raw" / constants.BOOK_DATA_FILENAME,
        config.DATA_DIR / constants.BOOK_DATA_FILENAME,
        Path("D:\\br\\data\\raw\\books.csv"),
    ]

    books_path = None
    for path in possible_paths:
        if path.exists():
            books_path = path
            break

    if books_path:
        try:
            books_df = pd.read_csv(books_path)
            print(f"Loaded books data from {books_path}")

            if constants.COL_AUTHOR_ID in books_df.columns:
                train_with_author = train_read.merge(
                    books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]],
                    on=constants.COL_BOOK_ID,
                    how='left'
                )
                train_with_author = train_with_author.dropna(subset=[constants.COL_AUTHOR_ID])

                if not train_with_author.empty:
                    author_agg = train_with_author.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(
                        ['mean']).reset_index()
                    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]
                    print(f"Computed aggregates for {len(author_agg)} authors")
        except Exception as e:
            print(f"Error loading books data: {e}")
    else:
        print(f"Books file not found. Tried paths: {possible_paths}")
        print(f"Current DATA_DIR: {config.DATA_DIR}")
        print(f"DATA_DIR exists: {config.DATA_DIR.exists()}")

    df = df.merge(user_agg, on=constants.COL_USER_ID, how='left')
    print(f"  Merged user aggregates: {len(user_agg)} users")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how='left')
    print(f"  Merged book aggregates: {len(book_agg)} books")

    if author_agg is not None and not author_agg.empty:
        if constants.COL_AUTHOR_ID not in df.columns and books_df is not None:
            df = df.merge(
                books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]],
                on=constants.COL_BOOK_ID,
                how='left'
            )

        if constants.COL_AUTHOR_ID in df.columns:
            df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how='left')

    user_na_mask = df[constants.F_USER_MEAN_RATING].isna()
    if user_na_mask.any():
        df.loc[user_na_mask, constants.F_USER_MEAN_RATING] = global_mean
        df.loc[user_na_mask, constants.F_USER_RATINGS_COUNT] = 0

    book_na_mask = df[constants.F_BOOK_MEAN_RATING].isna()
    if book_na_mask.any():
        df.loc[book_na_mask, constants.F_BOOK_MEAN_RATING] = global_mean * 1.05
        df.loc[book_na_mask, constants.F_BOOK_RATINGS_COUNT] = 0

    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        author_na_mask = df[constants.F_AUTHOR_MEAN_RATING].isna()
        if author_na_mask.any():
            df.loc[author_na_mask, constants.F_AUTHOR_MEAN_RATING] = global_mean

    print("Creating interaction features...")

    df['user_book_rating_diff'] = df[constants.F_USER_MEAN_RATING] - df[constants.F_BOOK_MEAN_RATING]

    df['user_book_rating_abs_diff'] = df['user_book_rating_diff'].abs()

    df['user_book_compatibility'] = 10 - df['user_book_rating_abs_diff']

    df['user_popularity_log'] = np.log1p(df[constants.F_USER_RATINGS_COUNT])
    df['book_popularity_log'] = np.log1p(df[constants.F_BOOK_RATINGS_COUNT])
    df['user_book_popularity_product'] = df['user_popularity_log'] * df['book_popularity_log']

    df['user_reliability'] = 1 - np.exp(-df[constants.F_USER_RATINGS_COUNT] / 5)

    df['book_reliability'] = 1 - np.exp(-df[constants.F_BOOK_RATINGS_COUNT] / 10)

    df['combined_reliability'] = df['user_reliability'] * 0.6 + df['book_reliability'] * 0.4

    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df['expected_rating'] = (
            df[constants.F_USER_MEAN_RATING] * 0.4 +
            df[constants.F_BOOK_MEAN_RATING] * 0.4 +
            df[constants.F_AUTHOR_MEAN_RATING] * 0.2
        )
    else:
        df['expected_rating'] = (
            df[constants.F_USER_MEAN_RATING] * 0.5 +
            df[constants.F_BOOK_MEAN_RATING] * 0.5
        )

    df['expected_rating'] = df['expected_rating'].clip(0, 10)

    feature_cols = [
        constants.F_USER_MEAN_RATING, constants.F_USER_RATINGS_COUNT,
        constants.F_BOOK_MEAN_RATING, constants.F_BOOK_RATINGS_COUNT,
        'user_book_rating_diff', 'user_book_rating_abs_diff',
        'user_book_compatibility', 'user_book_popularity_product',
        'user_reliability', 'book_reliability', 'combined_reliability',
        'expected_rating'
    ]

    # Добавляем author фичу, если есть
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        feature_cols.append(constants.F_AUTHOR_MEAN_RATING)

    num_features = len([c for c in feature_cols if c in df.columns])

    print(f" Added {num_features} safe aggregate features")
    print(f"   Global mean: {global_mean:.3f}")
    print(f"   Global std: {global_std:.3f}")
    print(f"   User coverage: {(~user_na_mask).mean():.2%}")
    print(f"   Book coverage: {(~book_na_mask).mean():.2%}")

    return df


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The DataFrame with the new 'book_genres_count' column.
    """
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds TF-IDF features from book descriptions.

    Trains a TF-IDF vectorizer only on training data descriptions to avoid
    data leakage. Applies the vectorizer to all books and merges the features.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for fitting the vectorizer.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with TF-IDF features added.
    """
    print("Adding text features (TF-IDF)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # Get unique books from train set
    train_books = train_df[constants.COL_BOOK_ID].unique()

    # Extract descriptions for training books only
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Check if vectorizer already exists (for prediction)
    if vectorizer_path.exists():
        print(f"Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        # Fit vectorizer on training descriptions only
        print("Fitting TF-IDF vectorizer on training descriptions...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        # Save vectorizer for use in prediction
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    # Transform all book descriptions
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Get descriptions in the same order as df[book_id]
    # Create a mapping book_id -> description
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )

    # Get descriptions for books in df (in the same order)
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")

    # Transform to TF-IDF features
    tfidf_matrix = vectorizer.transform(df_descriptions)

    # Convert sparse matrix to DataFrame
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    # Concatenate TF-IDF features with main DataFrame
    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(tfidf_feature_names)} TF-IDF features.")
    return df_with_tfidf


def add_nomic_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds Nomic embeddings from book descriptions.

    Extracts 768-dimensional embeddings using nomic-embed-text-v1.5 model.
    Embeddings are cached on disk to avoid recomputation on subsequent runs.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        _train_df (pd.DataFrame): The training portion (for consistency, not used for Nomic).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with Nomic embeddings added.
    """
    print("Adding text features (Nomic embeddings)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.NOMIC_EMBEDDINGS_FILENAME

    # Check if embeddings are already cached
    if embeddings_path.exists():
        print(f"Loading cached Nomic embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing Nomic embeddings (this may take a while)...")
        print(f"Using device: {config.NOMIC_DEVICE}")

        # Limit GPU memory usage to prevent OOM errors
        if config.NOMIC_DEVICE == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(config.NOMIC_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.NOMIC_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        # Load Nomic model using sentence-transformers
        # trust_remote_code=True is required for nomic-embed-text-v1.5
        model = SentenceTransformer(
            config.NOMIC_MODEL_NAME,
            device=config.NOMIC_DEVICE,
            trust_remote_code=True,
        )

        # Prepare descriptions: get unique book_id -> description mapping
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

        # Get unique books and their descriptions
        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()

        # Initialize embeddings dictionary
        embeddings_dict = {}

        # Process descriptions in batches
        num_batches = (len(descriptions) + config.NOMIC_BATCH_SIZE - 1) // config.NOMIC_BATCH_SIZE

        for batch_idx in tqdm(range(num_batches), desc="Processing Nomic batches", unit="batch"):
            start_idx = batch_idx * config.NOMIC_BATCH_SIZE
            end_idx = min(start_idx + config.NOMIC_BATCH_SIZE, len(descriptions))
            batch_descriptions = descriptions[start_idx:end_idx]
            batch_book_ids = book_ids[start_idx:end_idx]

            # Encode batch - sentence-transformers handles tokenization and pooling automatically
            batch_embeddings = model.encode(
                batch_descriptions,
                batch_size=len(batch_descriptions),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # Keep raw embeddings for feature engineering
            )

            # Store embeddings
            for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                embeddings_dict[book_id] = embedding

            # Small pause between batches to let GPU cool down and prevent overheating
            if config.NOMIC_DEVICE == "cuda":
                time.sleep(0.1)  # 100ms pause between batches

        # Save embeddings for future use
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved Nomic embeddings to {embeddings_path}")

    # Map embeddings to DataFrame rows by book_id
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()

    # Create embedding matrix
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            # Zero embedding for books without descriptions
            embeddings_list.append(np.zeros(config.NOMIC_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)

    # Create DataFrame with Nomic features
    nomic_feature_names = [f"nomic_{i}" for i in range(config.NOMIC_EMBEDDING_DIM)]
    nomic_df = pd.DataFrame(embeddings_array, columns=nomic_feature_names, index=df.index)

    # Concatenate Nomic features with main DataFrame
    df_with_nomic = pd.concat([df.reset_index(drop=True), nomic_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(nomic_feature_names)} Nomic features.")
    return df_with_nomic


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """Fills missing values using a defined strategy.

    Fills missing values for age, aggregated features, and categorical features
    to prepare the DataFrame for model training. Uses metrics from the training
    set (e.g., global mean) to fill NaNs.

    Args:
        df (pd.DataFrame): The DataFrame with missing values.
        train_df (pd.DataFrame): The training data, used for calculating fill metrics.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    print("Handling missing values...")

    # Calculate global mean from training data for filling
    global_mean = train_df[config.TARGET].mean()

    # Fill age with the median
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill aggregate features for "cold start" users/items (only if they exist)
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Fill TF-IDF features with 0 (for books without descriptions)
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    # Fill Nomic features with 0 (for books without descriptions)
    nomic_cols = [col for col in df.columns if col.startswith("nomic_")]
    for col in nomic_cols:
        df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    return df


def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    """Runs the full feature engineering pipeline.

    This function orchestrates the calls to add aggregate features (optional), genre
    features, text features (TF-IDF and Nomic), and handle missing values.

    Args:
        df (pd.DataFrame): The merged DataFrame from `data_processing`.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.
        include_aggregates (bool): If True, compute aggregate features. Defaults to False.
            Aggregates are typically computed separately during training to avoid data leakage.

    Returns:
        pd.DataFrame: The final DataFrame with all features engineered.
    """
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Aggregate features are computed separately during training to ensure
    # no data leakage from validation set timestamps
    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    df = add_nomic_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for CatBoost
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
