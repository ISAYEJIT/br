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
    Безопасные агрегатные фичи без leakage + ВСЕ улучшения
    """
    print("Adding enhanced aggregate features...")

    train_read = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    global_mean = train_read[config.TARGET].mean()

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

    print("  Adding smoothed aggregates...")

    user_smoothing_factor = 5
    user_agg[constants.F_USER_SMOOTHED_MEAN] = (
        (user_agg[constants.F_USER_MEAN_RATING] * user_agg[constants.F_USER_RATINGS_COUNT] +
         global_mean * user_smoothing_factor) /
        (user_agg[constants.F_USER_RATINGS_COUNT] + user_smoothing_factor)
    )

    book_smoothing_factor = 10
    book_agg[constants.F_BOOK_SMOOTHED_MEAN] = (
        (book_agg[constants.F_BOOK_MEAN_RATING] * book_agg[constants.F_BOOK_RATINGS_COUNT] +
         global_mean * book_smoothing_factor) /
        (book_agg[constants.F_BOOK_RATINGS_COUNT] + book_smoothing_factor)
    )

    print("  Adding percentile ranks...")

    user_agg['user_rating_percentile'] = user_agg[constants.F_USER_MEAN_RATING].rank(pct=True)
    book_agg['book_rating_percentile'] = book_agg[constants.F_BOOK_MEAN_RATING].rank(pct=True)
    user_agg['user_popularity_percentile'] = user_agg[constants.F_USER_RATINGS_COUNT].rank(pct=True)
    book_agg['book_popularity_percentile'] = book_agg[constants.F_BOOK_RATINGS_COUNT].rank(pct=True)

    df = df.merge(user_agg, on=constants.COL_USER_ID, how='left')
    print(f"  Merged user aggregates: {len(user_agg)} users")

    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how='left')
    print(f"  Merged book aggregates: {len(book_agg)} books")

    try:
        books_path = config.DATA_DIR / "raw" / constants.BOOK_DATA_FILENAME
        if not books_path.exists():
            books_path = config.DATA_DIR / constants.BOOK_DATA_FILENAME

        books_df = pd.read_csv(books_path)
        if 'author_id' in books_df.columns:
            books_df.rename(columns={'author_id': constants.COL_AUTHOR_ID}, inplace=True)

        train_with_author = train_read.merge(
            books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]],
            on=constants.COL_BOOK_ID,
            how='left'
        ).dropna(subset=[constants.COL_AUTHOR_ID])

        author_agg = train_with_author.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(
            ['mean', 'count']).reset_index()
        author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING, constants.F_AUTHOR_RATINGS_COUNT]

        author_agg[constants.F_AUTHOR_SMOOTHED_MEAN] = (
            (author_agg[constants.F_AUTHOR_MEAN_RATING] * author_agg[constants.F_AUTHOR_RATINGS_COUNT] +
             global_mean * 15) /
            (author_agg[constants.F_AUTHOR_RATINGS_COUNT] + 15)
        )

        if constants.COL_AUTHOR_ID not in df.columns:
            df = df.merge(books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]], on=constants.COL_BOOK_ID, how='left')

        df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how='left')
        print(f"  Computed aggregates for {len(author_agg)} authors")
    except Exception as e:
        print(f"  Warning: Could not load author aggregates: {e}")

    df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    df[constants.F_USER_SMOOTHED_MEAN] = df[constants.F_USER_SMOOTHED_MEAN].fillna(global_mean)
    df['user_rating_percentile'] = df['user_rating_percentile'].fillna(0.5)
    df['user_popularity_percentile'] = df['user_popularity_percentile'].fillna(0.5)

    df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean * 1.05)
    df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)
    df[constants.F_BOOK_SMOOTHED_MEAN] = df[constants.F_BOOK_SMOOTHED_MEAN].fillna(global_mean * 1.05)
    df['book_rating_percentile'] = df['book_rating_percentile'].fillna(0.5)
    df['book_popularity_percentile'] = df['book_popularity_percentile'].fillna(0.5)

    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)
        df[constants.F_AUTHOR_RATINGS_COUNT] = df[constants.F_AUTHOR_RATINGS_COUNT].fillna(0)
        df[constants.F_AUTHOR_SMOOTHED_MEAN] = df[constants.F_AUTHOR_SMOOTHED_MEAN].fillna(global_mean)

    print("  Adding stability features...")

    df['user_reliability_advanced'] = 1 - np.exp(-df[constants.F_USER_RATINGS_COUNT] / 8)
    df['book_reliability_advanced'] = 1 - np.exp(-df[constants.F_BOOK_RATINGS_COUNT] / 15)

    if constants.F_AUTHOR_RATINGS_COUNT in df.columns:
        df['author_reliability'] = 1 - np.exp(-df[constants.F_AUTHOR_RATINGS_COUNT] / 20)

    reliability_cols = ['user_reliability_advanced', 'book_reliability_advanced']
    if 'author_reliability' in df.columns:
        reliability_cols.append('author_reliability')

    df['combined_reliability_advanced'] = df[reliability_cols].mean(axis=1)

    print("Creating interaction features...")

    df['user_book_rating_diff'] = df[constants.F_USER_MEAN_RATING] - df[constants.F_BOOK_MEAN_RATING]
    df['user_book_rating_abs_diff'] = df['user_book_rating_diff'].abs()
    df['user_book_compatibility'] = 10 - df['user_book_rating_abs_diff']

    print("  Adding weighted interactions...")

    df['weighted_user_book_diff'] = (
        df[constants.F_USER_MEAN_RATING] * df['user_reliability_advanced'] -
        df[constants.F_BOOK_MEAN_RATING] * df['book_reliability_advanced']
    )

    total_reliability = df['user_reliability_advanced'] + df['book_reliability_advanced'] + 1e-10
    df['weighted_expected_rating'] = (
        df[constants.F_USER_MEAN_RATING] * (df['user_reliability_advanced'] / total_reliability) +
        df[constants.F_BOOK_MEAN_RATING] * (df['book_reliability_advanced'] / total_reliability)
    )

    print("  Adding advanced combinations...")

    df['percentile_compatibility'] = 1 - abs(df['user_rating_percentile'] - df['book_rating_percentile'])
    df['percentile_similarity'] = df['user_rating_percentile'] * df['book_rating_percentile']

    df['user_popularity_log'] = np.log1p(df[constants.F_USER_RATINGS_COUNT])
    df['book_popularity_log'] = np.log1p(df[constants.F_BOOK_RATINGS_COUNT])
    df['user_book_popularity_product'] = df['user_popularity_log'] * df['book_popularity_log']

    df['popularity_alignment'] = df['user_popularity_percentile'] * df['book_popularity_percentile']

    df['smoothed_user_book_diff'] = df[constants.F_USER_SMOOTHED_MEAN] - df[constants.F_BOOK_SMOOTHED_MEAN]
    df['smoothed_user_book_abs_diff'] = df['smoothed_user_book_diff'].abs()

    print("  Creating prediction ensemble...")

    has_author = constants.F_AUTHOR_MEAN_RATING in df.columns
    
    if has_author:
        df['expected_rating'] = (
            df[constants.F_USER_MEAN_RATING] * 0.4 +
            df[constants.F_BOOK_MEAN_RATING] * 0.4 +
            df[constants.F_AUTHOR_MEAN_RATING] * 0.2
        )
        df['expected_rating_smoothed'] = (
            df[constants.F_USER_SMOOTHED_MEAN] * 0.4 +
            df[constants.F_BOOK_SMOOTHED_MEAN] * 0.4 +
            df[constants.F_AUTHOR_SMOOTHED_MEAN] * 0.2
        )
    else:
        df['expected_rating'] = (
            df[constants.F_USER_MEAN_RATING] * 0.5 +
            df[constants.F_BOOK_MEAN_RATING] * 0.5
        )
        df['expected_rating_smoothed'] = (
            df[constants.F_USER_SMOOTHED_MEAN] * 0.5 +
            df[constants.F_BOOK_SMOOTHED_MEAN] * 0.5
        )

    df['final_ensemble_rating'] = (
        df['expected_rating'] * 0.3 +
        df['expected_rating_smoothed'] * 0.3 +
        df['weighted_expected_rating'] * 0.4
    ).clip(0, 10)

    if constants.COL_TIMESTAMP in df.columns:
        print("  Adding temporal features...")

        user_last_ts = train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].max().reset_index()
        user_last_ts.columns = [constants.COL_USER_ID, 'user_last_ts']

        df = df.merge(user_last_ts, on=constants.COL_USER_ID, how='left')
        df['days_since_last_read'] = (df[constants.COL_TIMESTAMP] - df['user_last_ts']).dt.days.abs()
        df['log_days_since_last'] = np.log1p(df['days_since_last_read'])
        df = df.drop('user_last_ts', axis=1)

        df['user_recently_active'] = (df['days_since_last_read'] <= 30).astype(int)
        df['active_reliable'] = df['user_recently_active'] * df['user_reliability_advanced']
    feature_cols = [
        constants.F_USER_MEAN_RATING, constants.F_USER_RATINGS_COUNT,
        constants.F_BOOK_MEAN_RATING, constants.F_BOOK_RATINGS_COUNT,
        constants.F_USER_SMOOTHED_MEAN, constants.F_BOOK_SMOOTHED_MEAN,
        'user_rating_percentile', 'book_rating_percentile',
        'user_popularity_percentile', 'book_popularity_percentile',
        'user_reliability_advanced', 'book_reliability_advanced',
        'combined_reliability_advanced',
        'user_book_rating_diff', 'user_book_rating_abs_diff',
        'user_book_compatibility', 'user_book_popularity_product',
        'weighted_user_book_diff', 'weighted_expected_rating',
        'percentile_compatibility', 'percentile_similarity',
        'smoothed_user_book_diff', 'smoothed_user_book_abs_diff',
        'popularity_alignment',
        'expected_rating', 'expected_rating_smoothed', 'final_ensemble_rating',
    ]

    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        feature_cols.extend([
            constants.F_AUTHOR_MEAN_RATING, constants.F_AUTHOR_RATINGS_COUNT,
            constants.F_AUTHOR_SMOOTHED_MEAN, 'author_reliability'
        ])

    if 'days_since_last_read' in df.columns:
        feature_cols.extend([
            'days_since_last_read', 'log_days_since_last',
            'user_recently_active', 'active_reliable'
        ])

    num_features = len([c for c in feature_cols if c in df.columns])
    print(f"✅ Added {num_features} enhanced aggregate features (global_mean={global_mean:.3f})")
    return df


def add_temporal_master_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    ВРЕМЕННЫЕ ФИЧИ - САМЫЙ БОЛЬШОЙ ПРИРОСТ МЕТРИКИ
    Добавляет 7 мощных временных фич без leakage
    """
    print("Adding TEMPORAL MASTER features...")

    if all(col in df.columns for col in [constants.COL_TIMESTAMP, constants.COL_PUBLICATION_YEAR]):
        df['timestamp_year'] = df[constants.COL_TIMESTAMP].dt.year
        df['book_age_at_reading'] = df['timestamp_year'] - df[constants.COL_PUBLICATION_YEAR]
        df['book_age_at_reading'] = df['book_age_at_reading'].clip(0, 150)
        df['log_book_age'] = np.log1p(df['book_age_at_reading'])
        df['book_age_category'] = pd.cut(
            df['book_age_at_reading'],
            bins=[0, 1, 5, 10, 20, 50, 150],
            labels=['new', 'recent', 'modern', 'established', 'classic', 'antique']
        )
        df = pd.get_dummies(df, columns=['book_age_category'], prefix='age_cat')

    if constants.COL_TIMESTAMP in train_df.columns:
        user_last_ts = train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].max().reset_index()
        user_last_ts.columns = [constants.COL_USER_ID, 'user_last_ts']

        df = df.merge(user_last_ts, on=constants.COL_USER_ID, how='left')

        if constants.COL_TIMESTAMP in df.columns:
            df['days_since_last_read'] = (df[constants.COL_TIMESTAMP] - df['user_last_ts']).dt.days.abs()
            df['log_days_since_last'] = np.log1p(df['days_since_last_read'])
            df['sqrt_days_since_last'] = np.sqrt(df['days_since_last_read'])
            df['user_recently_active_7d'] = (df['days_since_last_read'] <= 7).astype(int)
            df['user_recently_active_30d'] = (df['days_since_last_read'] <= 30).astype(int)
            df['user_inactive_90d'] = (df['days_since_last_read'] > 90).astype(int)

            df = df.drop('user_last_ts', axis=1)

    if constants.COL_TIMESTAMP in df.columns:
        df['reading_month'] = df[constants.COL_TIMESTAMP].dt.month
        df['reading_dayofweek'] = df[constants.COL_TIMESTAMP].dt.dayofweek
        df['reading_hour'] = df[constants.COL_TIMESTAMP].dt.hour

        df['reading_season'] = df['reading_month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        df = pd.get_dummies(df, columns=['reading_season'], prefix='season')

    print(
        f"  Added {len([c for c in df.columns if 'days_since' in c or 'age' in c or 'reading_' in c])} temporal features")
    return df


def add_smart_smoothing_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    УМНОЕ СГЛАЖИВАНИЕ - УЛУЧШАЕТ РЕДКИХ ПОЛЬЗОВАТЕЛЕЙ/КНИГИ
    Динамическая регуляризация на основе доверительных интервалов
    """
    print("Adding SMART SMOOTHING features...")

    train_read = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    global_mean = train_read[config.TARGET].mean()
    global_std = train_read[config.TARGET].std()

    user_stats = train_read.groupby(constants.COL_USER_ID)[config.TARGET].agg(['mean', 'count', 'std']).reset_index()
    user_stats.columns = [constants.COL_USER_ID, 'user_raw_mean', 'user_count', 'user_std']

    user_stats['user_smoothing_weight'] = 10 / (1 + np.log1p(user_stats['user_count']))
    user_stats['user_bayesian_mean'] = (
        (user_stats['user_raw_mean'] * user_stats['user_count'] +
         global_mean * user_stats['user_smoothing_weight']) /
        (user_stats['user_count'] + user_stats['user_smoothing_weight'])
    )

    def wilson_ci(mean, count, confidence=0.95):
        z = 1.96
        p = mean / 10
        denominator = 1 + z ** 2 / count
        centre = p + z ** 2 / (2 * count)
        half_width = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * count)) / count)
        lower = (centre - half_width) / denominator
        return lower * 10

    user_stats['user_wilson_lower'] = user_stats.apply(
        lambda x: wilson_ci(x['user_raw_mean'], x['user_count']), axis=1
    )

    book_stats = train_read.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(['mean', 'count', 'std']).reset_index()
    book_stats.columns = [constants.COL_BOOK_ID, 'book_raw_mean', 'book_count', 'book_std']

    book_stats['book_smoothing_weight'] = 20 / (1 + np.log1p(book_stats['book_count']))
    book_stats['book_bayesian_mean'] = (
        (book_stats['book_raw_mean'] * book_stats['book_count'] +
         global_mean * book_stats['book_smoothing_weight']) /
        (book_stats['book_count'] + book_stats['book_smoothing_weight'])
    )

    book_stats['book_wilson_lower'] = book_stats.apply(
        lambda x: wilson_ci(x['book_raw_mean'], x['book_count']), axis=1
    )

    df = df.merge(
        user_stats[[constants.COL_USER_ID, 'user_bayesian_mean', 'user_wilson_lower', 'user_smoothing_weight']],
        on=constants.COL_USER_ID, how='left'
    )

    df = df.merge(
        book_stats[[constants.COL_BOOK_ID, 'book_bayesian_mean', 'book_wilson_lower', 'book_smoothing_weight']],
        on=constants.COL_BOOK_ID, how='left'
    )

    df['user_bayesian_mean'] = df['user_bayesian_mean'].fillna(global_mean)
    df['user_wilson_lower'] = df['user_wilson_lower'].fillna(global_mean * 0.9)
    df['book_bayesian_mean'] = df['book_bayesian_mean'].fillna(global_mean)
    df['book_wilson_lower'] = df['book_wilson_lower'].fillna(global_mean * 0.9)

    df['smoothed_user_book_diff'] = df['user_bayesian_mean'] - df['book_bayesian_mean']
    df['conservative_estimate'] = (df['user_wilson_lower'] + df['book_wilson_lower']) / 2

    print(
        f"  Added {len([c for c in df.columns if 'bayesian' in c or 'wilson' in c or 'smoothing' in c])} smoothing features")
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

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    train_books = train_df[constants.COL_BOOK_ID].unique()

    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    if vectorizer_path.exists():
        print(f"Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("Fitting TF-IDF vectorizer on training descriptions...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )

    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")

    tfidf_matrix = vectorizer.transform(df_descriptions)

    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

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

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.NOMIC_EMBEDDINGS_FILENAME

    if embeddings_path.exists():
        print(f"Loading cached Nomic embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing Nomic embeddings (this may take a while)...")
        print(f"Using device: {config.NOMIC_DEVICE}")

        if config.NOMIC_DEVICE == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(config.NOMIC_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.NOMIC_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        model = SentenceTransformer(
            config.NOMIC_MODEL_NAME,
            device=config.NOMIC_DEVICE,
            trust_remote_code=True,
        )

        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()

        embeddings_dict = {}

        num_batches = (len(descriptions) + config.NOMIC_BATCH_SIZE - 1) // config.NOMIC_BATCH_SIZE

        for batch_idx in tqdm(range(num_batches), desc="Processing Nomic batches", unit="batch"):
            start_idx = batch_idx * config.NOMIC_BATCH_SIZE
            end_idx = min(start_idx + config.NOMIC_BATCH_SIZE, len(descriptions))
            batch_descriptions = descriptions[start_idx:end_idx]
            batch_book_ids = book_ids[start_idx:end_idx]

            batch_embeddings = model.encode(
                batch_descriptions,
                batch_size=len(batch_descriptions),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False, #???????????????????????????
            )

            for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                embeddings_dict[book_id] = embedding

            if config.NOMIC_DEVICE == "cuda":
                time.sleep(0.1)

        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved Nomic embeddings to {embeddings_path}")

    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()

    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            embeddings_list.append(np.zeros(config.NOMIC_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)

    nomic_feature_names = [f"nomic_{i}" for i in range(config.NOMIC_EMBEDDING_DIM)]
    nomic_df = pd.DataFrame(embeddings_array, columns=nomic_feature_names, index=df.index)

    df_with_nomic = pd.concat([df.reset_index(drop=True), nomic_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(nomic_feature_names)} Nomic features.")
    return df_with_nomic


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
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

    global_mean = train_df[config.TARGET].mean()

    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

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

    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    nomic_cols = [col for col in df.columns if col.startswith("nomic_")]
    for col in nomic_cols:
        df[col] = df[col].fillna(0.0)

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
    df = add_temporal_master_features(df, train_df)
    df = add_smart_smoothing_features(df, train_df)
    df = add_text_features(df, train_df, descriptions_df)
    df = add_nomic_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for CatBoost
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
