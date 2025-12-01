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
    global_std = train_read[config.TARGET].std()

    # ========== БАЗОВЫЕ АГРЕГАТЫ ==========
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

    # ========== УЛУЧШЕНИЕ 1: SMOOTHED AGGREGATES ==========
    print("  Adding smoothed aggregates...")

    # Smoothed user mean (Bayesian average)
    user_smoothing_factor = 5
    user_agg[constants.F_USER_SMOOTHED_MEAN] = (
        (user_agg[constants.F_USER_MEAN_RATING] * user_agg[constants.F_USER_RATINGS_COUNT] +
         global_mean * user_smoothing_factor) /
        (user_agg[constants.F_USER_RATINGS_COUNT] + user_smoothing_factor)
    )

    # Smoothed book mean
    book_smoothing_factor = 10
    book_agg[constants.F_BOOK_SMOOTHED_MEAN] = (
        (book_agg[constants.F_BOOK_MEAN_RATING] * book_agg[constants.F_BOOK_RATINGS_COUNT] +
         global_mean * book_smoothing_factor) /
        (book_agg[constants.F_BOOK_RATINGS_COUNT] + book_smoothing_factor)
    )

    # ========== УЛУЧШЕНИЕ 2: PERCENTILE RANKS ==========
    print("  Adding percentile ranks...")

    # User rating percentile
    user_agg['user_rating_percentile'] = user_agg[constants.F_USER_MEAN_RATING].rank(pct=True)

    # Book rating percentile
    book_agg['book_rating_percentile'] = book_agg[constants.F_BOOK_MEAN_RATING].rank(pct=True)

    # Popularity percentiles
    user_agg['user_popularity_percentile'] = user_agg[constants.F_USER_RATINGS_COUNT].rank(pct=True)
    book_agg['book_popularity_percentile'] = book_agg[constants.F_BOOK_RATINGS_COUNT].rank(pct=True)

    # ========== МЕРЖИМ БАЗОВЫЕ ФИЧИ ==========
    df = df.merge(user_agg, on=constants.COL_USER_ID, how='left')
    print(f"  Merged user aggregates: {len(user_agg)} users")

    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how='left')
    print(f"  Merged book aggregates: {len(book_agg)} books")

    # ========== AUTHOR AGGREGATES (исправленная версия) ==========
    author_agg = None
    try:
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
            books_df = pd.read_csv(books_path)
            print(f"  Loaded books data from {books_path}")

            # Проверяем, есть ли author_id в данных
            if 'author_id' in books_df.columns:
                # Переименовываем для консистентности
                books_df = books_df.rename(columns={'author_id': constants.COL_AUTHOR_ID})

            if constants.COL_AUTHOR_ID in books_df.columns:
                train_with_author = train_read.merge(
                    books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]],
                    on=constants.COL_BOOK_ID,
                    how='left'
                )
                train_with_author = train_with_author.dropna(subset=[constants.COL_AUTHOR_ID])

                if not train_with_author.empty:
                    author_agg = train_with_author.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(
                        ['mean', 'count']).reset_index()
                    author_agg.columns = [
                        constants.COL_AUTHOR_ID,
                        constants.F_AUTHOR_MEAN_RATING,
                        constants.F_AUTHOR_RATINGS_COUNT
                    ]

                    # Smoothed author mean
                    author_smoothing_factor = 15
                    author_agg[constants.F_AUTHOR_SMOOTHED_MEAN] = (
                        (author_agg[constants.F_AUTHOR_MEAN_RATING] * author_agg[constants.F_AUTHOR_RATINGS_COUNT] +
                         global_mean * author_smoothing_factor) /
                        (author_agg[constants.F_AUTHOR_RATINGS_COUNT] + author_smoothing_factor)
                    )

                    print(f"  Computed aggregates for {len(author_agg)} authors")

                    # Добавляем author_id в df если его нет
                    if constants.COL_AUTHOR_ID not in df.columns:
                        df = df.merge(
                            books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]],
                            on=constants.COL_BOOK_ID,
                            how='left'
                        )

                    # Мержим авторские агрегаты
                    df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how='left')
    except Exception as e:
        print(f"  Warning: Could not load author aggregates: {e}")
        print(f"  Error type: {type(e).__name__}")

    # ========== ЗАПОЛНЕНИЕ ПРОПУСКОВ ==========
    user_na_mask = df[constants.F_USER_MEAN_RATING].isna()
    if user_na_mask.any():
        df.loc[user_na_mask, constants.F_USER_MEAN_RATING] = global_mean
        df.loc[user_na_mask, constants.F_USER_RATINGS_COUNT] = 0
        df.loc[user_na_mask, constants.F_USER_SMOOTHED_MEAN] = global_mean
        df.loc[user_na_mask, 'user_rating_percentile'] = 0.5
        df.loc[user_na_mask, 'user_popularity_percentile'] = 0.5

    book_na_mask = df[constants.F_BOOK_MEAN_RATING].isna()
    if book_na_mask.any():
        df.loc[book_na_mask, constants.F_BOOK_MEAN_RATING] = global_mean * 1.05
        df.loc[book_na_mask, constants.F_BOOK_RATINGS_COUNT] = 0
        df.loc[book_na_mask, constants.F_BOOK_SMOOTHED_MEAN] = global_mean * 1.05
        df.loc[book_na_mask, 'book_rating_percentile'] = 0.5
        df.loc[book_na_mask, 'book_popularity_percentile'] = 0.5

    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        author_na_mask = df[constants.F_AUTHOR_MEAN_RATING].isna()
        if author_na_mask.any():
            df.loc[author_na_mask, constants.F_AUTHOR_MEAN_RATING] = global_mean
            df.loc[author_na_mask, constants.F_AUTHOR_RATINGS_COUNT] = 0
            if constants.F_AUTHOR_SMOOTHED_MEAN in df.columns:
                df.loc[author_na_mask, constants.F_AUTHOR_SMOOTHED_MEAN] = global_mean

    # ========== УЛУЧШЕНИЕ 3: STABILITY & RELIABILITY ==========
    print("  Adding stability features...")

    # User reliability (чем больше оценок, тем надежнее)
    df['user_reliability_advanced'] = 1 - np.exp(-df[constants.F_USER_RATINGS_COUNT] / 8)

    # Book reliability
    df['book_reliability_advanced'] = 1 - np.exp(-df[constants.F_BOOK_RATINGS_COUNT] / 15)

    # Author reliability (если есть)
    if constants.F_AUTHOR_RATINGS_COUNT in df.columns:
        df['author_reliability'] = 1 - np.exp(-df[constants.F_AUTHOR_RATINGS_COUNT] / 20)

    # Combined reliability
    reliability_cols = ['user_reliability_advanced', 'book_reliability_advanced']
    if 'author_reliability' in df.columns:
        reliability_cols.append('author_reliability')

    df['combined_reliability_advanced'] = df[reliability_cols].mean(axis=1)

    # ========== БАЗОВЫЕ INTERACTION FEATURES ==========
    print("Creating interaction features...")

    df['user_book_rating_diff'] = df[constants.F_USER_MEAN_RATING] - df[constants.F_BOOK_MEAN_RATING]
    df['user_book_rating_abs_diff'] = df['user_book_rating_diff'].abs()
    df['user_book_compatibility'] = 10 - df['user_book_rating_abs_diff']

    # ========== УЛУЧШЕНИЕ 4: WEIGHTED INTERACTIONS ==========
    print("  Adding weighted interactions...")

    # Weighted differences using reliability
    df['weighted_user_book_diff'] = (
        df[constants.F_USER_MEAN_RATING] * df['user_reliability_advanced'] -
        df[constants.F_BOOK_MEAN_RATING] * df['book_reliability_advanced']
    )

    # Weighted prediction
    if 'user_reliability_advanced' in df.columns and 'book_reliability_advanced' in df.columns:
        total_reliability = df['user_reliability_advanced'] + df['book_reliability_advanced'] + 1e-10
        df['weighted_expected_rating'] = (
            df[constants.F_USER_MEAN_RATING] * (df['user_reliability_advanced'] / total_reliability) +
            df[constants.F_BOOK_MEAN_RATING] * (df['book_reliability_advanced'] / total_reliability)
        )

    # ========== УЛУЧШЕНИЕ 5: ADVANCED COMBINATIONS ==========
    print("  Adding advanced combinations...")

    # Percentile compatibility
    if 'user_rating_percentile' in df.columns and 'book_rating_percentile' in df.columns:
        df['percentile_compatibility'] = 1 - abs(df['user_rating_percentile'] - df['book_rating_percentile'])
        df['percentile_similarity'] = df['user_rating_percentile'] * df['book_rating_percentile']

    # Popularity interactions
    df['user_popularity_log'] = np.log1p(df[constants.F_USER_RATINGS_COUNT])
    df['book_popularity_log'] = np.log1p(df[constants.F_BOOK_RATINGS_COUNT])
    df['user_book_popularity_product'] = df['user_popularity_log'] * df['book_popularity_log']

    if 'user_popularity_percentile' in df.columns and 'book_popularity_percentile' in df.columns:
        df['popularity_alignment'] = df['user_popularity_percentile'] * df['book_popularity_percentile']

    # Smoothed interactions
    df['smoothed_user_book_diff'] = df[constants.F_USER_SMOOTHED_MEAN] - df[constants.F_BOOK_SMOOTHED_MEAN]
    df['smoothed_user_book_abs_diff'] = df['smoothed_user_book_diff'].abs()

    # ========== FINAL PREDICTION ENSEMBLE ==========
    print("  Creating prediction ensemble...")

    # Base expected rating
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

    # Smoothed expected rating
    if constants.F_AUTHOR_SMOOTHED_MEAN in df.columns:
        df['expected_rating_smoothed'] = (
            df[constants.F_USER_SMOOTHED_MEAN] * 0.4 +
            df[constants.F_BOOK_SMOOTHED_MEAN] * 0.4 +
            df[constants.F_AUTHOR_SMOOTHED_MEAN] * 0.2
        )
    else:
        df['expected_rating_smoothed'] = (
            df[constants.F_USER_SMOOTHED_MEAN] * 0.5 +
            df[constants.F_BOOK_SMOOTHED_MEAN] * 0.5
        )

    # Reliability-adjusted rating
    if 'weighted_expected_rating' in df.columns:
        df['final_ensemble_rating'] = (
            df['expected_rating'] * 0.3 +
            df['expected_rating_smoothed'] * 0.3 +
            df['weighted_expected_rating'] * 0.4
        )
    else:
        df['final_ensemble_rating'] = (
            df['expected_rating'] * 0.5 +
            df['expected_rating_smoothed'] * 0.5
        )

    df['final_ensemble_rating'] = df['final_ensemble_rating'].clip(0, 10)

    # ========== TEMPORAL FEATURES (если есть timestamp) ==========
    if constants.COL_TIMESTAMP in df.columns:
        print("  Adding temporal features...")

        # Дни с последнего чтения (используем train_df для вычислений)
        user_last_ts = train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].max().reset_index()
        user_last_ts.columns = [constants.COL_USER_ID, 'user_last_ts']

        df = df.merge(user_last_ts, on=constants.COL_USER_ID, how='left')
        df['days_since_last_read'] = (df[constants.COL_TIMESTAMP] - df['user_last_ts']).dt.days.abs()
        df['log_days_since_last'] = np.log1p(df['days_since_last_read'])
        df = df.drop('user_last_ts', axis=1)

        # Активность пользователя
        df['user_recently_active'] = (df['days_since_last_read'] <= 30).astype(int)

        # Взаимодействие с надежностью
        df['active_reliable'] = df['user_recently_active'] * df['user_reliability_advanced']

    # ========== FINAL STATS ==========
    feature_cols = [
        # Базовые
        constants.F_USER_MEAN_RATING, constants.F_USER_RATINGS_COUNT,
        constants.F_BOOK_MEAN_RATING, constants.F_BOOK_RATINGS_COUNT,

        # Smoothed
        constants.F_USER_SMOOTHED_MEAN, constants.F_BOOK_SMOOTHED_MEAN,

        # Percentiles
        'user_rating_percentile', 'book_rating_percentile',
        'user_popularity_percentile', 'book_popularity_percentile',

        # Reliability
        'user_reliability_advanced', 'book_reliability_advanced',
        'combined_reliability_advanced',

        # Interactions
        'user_book_rating_diff', 'user_book_rating_abs_diff',
        'user_book_compatibility', 'user_book_popularity_product',
        'weighted_user_book_diff', 'weighted_expected_rating',
        'percentile_compatibility', 'percentile_similarity',
        'smoothed_user_book_diff', 'smoothed_user_book_abs_diff',
        'popularity_alignment',

        # Predictions
        'expected_rating', 'expected_rating_smoothed', 'final_ensemble_rating',
    ]

    # Добавляем author фичи если есть
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        feature_cols.extend([
            constants.F_AUTHOR_MEAN_RATING, constants.F_AUTHOR_RATINGS_COUNT,
            constants.F_AUTHOR_SMOOTHED_MEAN, 'author_reliability'
        ])

    # Добавляем temporal фичи если есть
    if 'days_since_last_read' in df.columns:
        feature_cols.extend([
            'days_since_last_read', 'log_days_since_last',
            'user_recently_active', 'active_reliable'
        ])

    num_features = len([c for c in feature_cols if c in df.columns])

    print(f"✅ Added {num_features} enhanced aggregate features")
    print(f"   Global mean: {global_mean:.3f}")
    print(f"   Global std: {global_std:.3f}")
    print(f"   User coverage: {(~user_na_mask).mean():.2%}")
    print(f"   Book coverage: {(~book_na_mask).mean():.2%}")

    if 'author_reliability' in df.columns:
        print(f"   Author coverage: {(~df[constants.F_AUTHOR_MEAN_RATING].isna()).mean():.2%}")

    return df


def add_temporal_master_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    ВРЕМЕННЫЕ ФИЧИ - САМЫЙ БОЛЬШОЙ ПРИРОСТ МЕТРИКИ
    Добавляет 7 мощных временных фич без leakage
    """
    print("Adding TEMPORAL MASTER features...")

    # 1. Возраст книги на момент чтения (ТОП-1 фича)
    if all(col in df.columns for col in [constants.COL_TIMESTAMP, constants.COL_PUBLICATION_YEAR]):
        df['timestamp_year'] = df[constants.COL_TIMESTAMP].dt.year
        df['book_age_at_reading'] = df['timestamp_year'] - df[constants.COL_PUBLICATION_YEAR]
        df['book_age_at_reading'] = df['book_age_at_reading'].clip(0, 150)

        # Логарифмированный возраст
        df['log_book_age'] = np.log1p(df['book_age_at_reading'])

        # Возрастные категории
        df['book_age_category'] = pd.cut(
            df['book_age_at_reading'],
            bins=[0, 1, 5, 10, 20, 50, 150],
            labels=['new', 'recent', 'modern', 'established', 'classic', 'antique']
        )
        df = pd.get_dummies(df, columns=['book_age_category'], prefix='age_cat')

    # 2. Дни с последнего чтения пользователя (ТОП-2 фича)
    if constants.COL_TIMESTAMP in train_df.columns:
        # Последняя активность пользователя в train
        user_last_ts = train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].max().reset_index()
        user_last_ts.columns = [constants.COL_USER_ID, 'user_last_ts']

        df = df.merge(user_last_ts, on=constants.COL_USER_ID, how='left')

        if constants.COL_TIMESTAMP in df.columns:
            # Базовый признак
            df['days_since_last_read'] = (df[constants.COL_TIMESTAMP] - df['user_last_ts']).dt.days.abs()

            # Трансформированные версии
            df['log_days_since_last'] = np.log1p(df['days_since_last_read'])
            df['sqrt_days_since_last'] = np.sqrt(df['days_since_last_read'])

            # Активность пользователя
            df['user_recently_active_7d'] = (df['days_since_last_read'] <= 7).astype(int)
            df['user_recently_active_30d'] = (df['days_since_last_read'] <= 30).astype(int)
            df['user_inactive_90d'] = (df['days_since_last_read'] > 90).astype(int)

            df = df.drop('user_last_ts', axis=1)

    # 3. Время года / день недели (сезонность)
    if constants.COL_TIMESTAMP in df.columns:
        df['reading_month'] = df[constants.COL_TIMESTAMP].dt.month
        df['reading_dayofweek'] = df[constants.COL_TIMESTAMP].dt.dayofweek
        df['reading_hour'] = df[constants.COL_TIMESTAMP].dt.hour

        # Сезонные категории
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

    # 1. Bayesian Smoothing с динамическим weight
    user_stats = train_read.groupby(constants.COL_USER_ID)[config.TARGET].agg(['mean', 'count', 'std']).reset_index()
    user_stats.columns = [constants.COL_USER_ID, 'user_raw_mean', 'user_count', 'user_std']

    # Динамический smoothing factor: чем меньше count, тем сильнее сглаживание
    user_stats['user_smoothing_weight'] = 10 / (1 + np.log1p(user_stats['user_count']))
    user_stats['user_bayesian_mean'] = (
        (user_stats['user_raw_mean'] * user_stats['user_count'] +
         global_mean * user_stats['user_smoothing_weight']) /
        (user_stats['user_count'] + user_stats['user_smoothing_weight'])
    )

    # 2. Wilson Confidence Interval для пользователей
    # (дает более консервативные оценки для редких случаев)
    def wilson_ci(mean, count, confidence=0.95):
        z = 1.96  # для 95% доверительного интервала
        p = mean / 10  # нормализуем к [0,1]
        denominator = 1 + z ** 2 / count
        centre = p + z ** 2 / (2 * count)
        half_width = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * count)) / count)
        lower = (centre - half_width) / denominator
        return lower * 10  # возвращаем к шкале [0,10]

    user_stats['user_wilson_lower'] = user_stats.apply(
        lambda x: wilson_ci(x['user_raw_mean'], x['user_count']), axis=1
    )

    # 3. Аналогично для книг
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

    # 4. Мержим smoothed фичи
    df = df.merge(
        user_stats[[constants.COL_USER_ID, 'user_bayesian_mean', 'user_wilson_lower', 'user_smoothing_weight']],
        on=constants.COL_USER_ID, how='left'
    )

    df = df.merge(
        book_stats[[constants.COL_BOOK_ID, 'book_bayesian_mean', 'book_wilson_lower', 'book_smoothing_weight']],
        on=constants.COL_BOOK_ID, how='left'
    )

    # 5. Заполняем пропуски
    df['user_bayesian_mean'] = df['user_bayesian_mean'].fillna(global_mean)
    df['user_wilson_lower'] = df['user_wilson_lower'].fillna(global_mean * 0.9)
    df['book_bayesian_mean'] = df['book_bayesian_mean'].fillna(global_mean)
    df['book_wilson_lower'] = df['book_wilson_lower'].fillna(global_mean * 0.9)

    # 6. Создаем комбинированные фичи
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
    #df = add_temporal_master_features(df, train_df)
    #df = add_smart_smoothing_features(df, train_df)
    df = add_text_features(df, train_df, descriptions_df)
    df = add_nomic_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for CatBoost
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
