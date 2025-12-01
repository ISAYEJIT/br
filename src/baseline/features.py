"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from scipy import sparse
from scipy.sparse.linalg import spsolve
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from . import config, constants


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features.

    Uses the training data to compute mean ratings and interaction counts
    to prevent data leakage from the test set.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion of the data for calculations.

    Returns:
        pd.DataFrame: The DataFrame with new aggregate features.
    """
    print("Adding aggregate features...")

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")


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


def add_ubcf_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_similar_users = 50,
    use_timestamp_filtering: bool = False,
) -> pd.DataFrame:
    """Adds User-Based Collaborative Filtering features.
    """
    print("Adding improved UBCF features (no feature cache)...")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    similarity_cache_path = config.MODEL_DIR / "ubcf_similarity_only.pkl"

    # Step 1: Load or compute user similarities (ONLY similarities, not features)
    if similarity_cache_path.exists():
        print(f"Loading user similarity cache from {similarity_cache_path}")
        cache = joblib.load(similarity_cache_path)
        top_similar_users_dict = cache["top_similar_users"]
        print(f"Loaded similarities for {len(top_similar_users_dict)} users")
    else:
        print("Computing user similarities from training data...")

        # Only use has_read=1 ratings for similarity
        rated_train = train_df[train_df[constants.COL_HAS_READ] == 1].copy()

        # Build user-book rating matrix
        print("Building user-book matrix...")
        user_book_matrix = rated_train.pivot_table(
            index=constants.COL_USER_ID,
            columns=constants.COL_BOOK_ID,
            values=config.TARGET,
            fill_value=0
        )
        print(f"Matrix shape: {user_book_matrix.shape}")

        # Compute cosine similarity between users
        print("Computing cosine similarities...")
        user_similarity = cosine_similarity(user_book_matrix.values)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_book_matrix.index,
            columns=user_book_matrix.index
        )

        # Store only top-K similar users for each user
        print(f"Extracting top {n_similar_users} similar users...")
        top_similar_users_dict = {}
        for user_id in tqdm(user_similarity_df.index, desc="Processing users"):
            user_sims = user_similarity_df.loc[user_id].sort_values(ascending=False)
            # Exclude self-similarity
            user_sims = user_sims[user_sims.index != user_id]
            # Store top K as dict {similar_user_id: similarity_score}
            top_similar_users_dict[user_id] = user_sims.head(n_similar_users).to_dict()

        # Save only similarity information
        print(f"Saving similarity cache to {similarity_cache_path}")
        joblib.dump(
            {"top_similar_users": top_similar_users_dict},
            similarity_cache_path
        )
        print("Similarity cache saved")

    # Step 2: Prepare rating lookup for dynamic feature computation
    print("Preparing rating lookup...")
    rated_train = train_df[train_df[constants.COL_HAS_READ] == 1].copy()

    if use_timestamp_filtering and constants.COL_TIMESTAMP in rated_train.columns:
        # Convert timestamp to datetime for filtering
        rated_train[constants.COL_TIMESTAMP] = pd.to_datetime(rated_train[constants.COL_TIMESTAMP])
        has_timestamps = True
        print("Temporal filtering enabled - will use only past ratings")
    else:
        has_timestamps = False
        print("Temporal filtering disabled - using all training ratings")

    # Create efficient lookup: (user_id, book_id) -> rating
    rating_lookup = {}
    timestamp_lookup = {}

    for _, row in rated_train.iterrows():
        key = (row[constants.COL_USER_ID], row[constants.COL_BOOK_ID])
        rating_lookup[key] = row[config.TARGET]
        if has_timestamps:
            timestamp_lookup[key] = row[constants.COL_TIMESTAMP]

    # Step 3: Compute features dynamically for ALL rows
    print(f"Computing UBCF features for {len(df)} rows...")

    # Convert df timestamps if needed
    if has_timestamps and constants.COL_TIMESTAMP in df.columns:
        df_timestamps = pd.to_datetime(df[constants.COL_TIMESTAMP])
    else:
        df_timestamps = [None] * len(df)

    ubcf_scores = []
    ubcf_counts = []
    ubcf_stds = []  # NEW: standard deviation of similar users' ratings
    ubcf_max_sim = []  # NEW: max similarity score among users who rated this book

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="UBCF features")):
        user_id = row[constants.COL_USER_ID]
        book_id = row[constants.COL_BOOK_ID]
        current_timestamp = df_timestamps[idx]

        score = np.nan
        count = 0
        ratings_list = []
        max_similarity = 0.0

        # Check if we have similar users for this user
        if user_id in top_similar_users_dict:
            similar_users = top_similar_users_dict[user_id]

            # Collect ratings from similar users
            for similar_user_id, similarity_score in similar_users.items():
                lookup_key = (similar_user_id, book_id)

                # Check if similar user rated this book
                if lookup_key in rating_lookup:
                    rating = rating_lookup[lookup_key]

                    # Temporal filtering: only use past ratings
                    if has_timestamps and current_timestamp is not None:
                        rating_timestamp = timestamp_lookup.get(lookup_key)
                        if rating_timestamp is not None and rating_timestamp >= current_timestamp:
                            continue  # Skip future ratings

                    ratings_list.append(rating)
                    count += 1
                    max_similarity = max(max_similarity, similarity_score)

            # Compute weighted average if we have ratings
            if ratings_list:
                # Simple mean (you can also do weighted by similarity)
                score = np.mean(ratings_list)

        # Compute standard deviation (measure of agreement)
        std = np.std(ratings_list) if len(ratings_list) > 1 else 0.0

        ubcf_scores.append(score)
        ubcf_counts.append(count)
        ubcf_stds.append(std)
        ubcf_max_sim.append(max_similarity if count > 0 else 0.0)

    # Add features to dataframe
    df[constants.F_UBCF_SCORE] = ubcf_scores
    df[constants.F_UBCF_COUNT] = ubcf_counts
    df[constants.F_UBCF_STD] = ubcf_stds
    df[constants.F_UBCF_MAX_SIM] = ubcf_max_sim

    print(f"UBCF features added:")
    print(f"  - {constants.F_UBCF_SCORE}: avg rating from similar users")
    print(f"  - {constants.F_UBCF_COUNT}: number of similar users who rated")
    print(f"  - {constants.F_UBCF_STD}: std of ratings (agreement measure)")
    print(f"  - {constants.F_UBCF_MAX_SIM}: max similarity score")
    print(f"Coverage: {(~pd.Series(ubcf_scores).isna()).mean():.2%} of rows have UBCF scores")

    return df


def add_ibcf_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_similar_items = 50,
    use_timestamp_filtering: bool = True,
) -> pd.DataFrame:
    """Adds Item-Based Collaborative Filtering features.
    """
    print("Adding Item-Based Collaborative Filtering features...")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ibcf_cache_path = config.MODEL_DIR / "ibcf_similarity.pkl"

    # Step 1: Load or compute item similarities
    if ibcf_cache_path.exists():
        print(f"Loading item similarity cache from {ibcf_cache_path}")
        cache = joblib.load(ibcf_cache_path)
        top_similar_items_dict = cache["top_similar_items"]
        print(f"Loaded similarities for {len(top_similar_items_dict)} books")
    else:
        print("Computing item-item similarities...")

        # Only use has_read=1 ratings
        rated_train = train_df[train_df[constants.COL_HAS_READ] == 1].copy()

        # Build item-user rating matrix (transposed from UBCF)
        print("Building book-user matrix...")
        book_user_matrix = rated_train.pivot_table(
            index=constants.COL_BOOK_ID,
            columns=constants.COL_USER_ID,
            values=config.TARGET,
            fill_value=0
        )
        print(f"Matrix shape: {book_user_matrix.shape}")

        # Compute cosine similarity between books
        print("Computing item similarities...")
        item_similarity = cosine_similarity(book_user_matrix.values)
        item_similarity_df = pd.DataFrame(
            item_similarity,
            index=book_user_matrix.index,
            columns=book_user_matrix.index
        )

        # Store top-K similar items
        print(f"Extracting top {n_similar_items} similar items...")
        top_similar_items_dict = {}
        for book_id in tqdm(item_similarity_df.index, desc="Processing books"):
            item_sims = item_similarity_df.loc[book_id].sort_values(ascending=False)
            item_sims = item_sims[item_sims.index != book_id]
            top_similar_items_dict[book_id] = item_sims.head(n_similar_items).to_dict()

        print(f"Saving IBCF cache to {ibcf_cache_path}")
        joblib.dump({"top_similar_items": top_similar_items_dict}, ibcf_cache_path)

    # Step 2: Prepare user rating lookup with timestamps
    print("Preparing user rating lookup...")
    rated_train = train_df[train_df[constants.COL_HAS_READ] == 1].copy()

    if use_timestamp_filtering and constants.COL_TIMESTAMP in rated_train.columns:
        # Convert timestamp to datetime for filtering
        rated_train[constants.COL_TIMESTAMP] = pd.to_datetime(rated_train[constants.COL_TIMESTAMP])
        has_timestamps = True
        print("Temporal filtering enabled - will use only past ratings")
    else:
        has_timestamps = False
        print("Temporal filtering disabled - using all training ratings")

    # Create efficient lookup: (user_id, book_id) -> rating and timestamp
    user_ratings_lookup = {}
    timestamp_lookup = {}

    for _, row in rated_train.iterrows():
        key = (row[constants.COL_USER_ID], row[constants.COL_BOOK_ID])
        user_ratings_lookup[key] = row[config.TARGET]
        if has_timestamps:
            timestamp_lookup[key] = row[constants.COL_TIMESTAMP]

    # Step 3: Compute IBCF features with temporal filtering
    print(f"Computing IBCF features for {len(df)} rows...")

    # Convert df timestamps if needed
    if has_timestamps and constants.COL_TIMESTAMP in df.columns:
        df_timestamps = pd.to_datetime(df[constants.COL_TIMESTAMP])
    else:
        df_timestamps = [None] * len(df)

    ibcf_scores = []
    ibcf_counts = []
    ibcf_stds = []
    ibcf_max_sim = []

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="IBCF features")):
        user_id = row[constants.COL_USER_ID]
        book_id = row[constants.COL_BOOK_ID]
        current_timestamp = df_timestamps[idx]

        score = np.nan
        count = 0
        ratings_list = []
        max_similarity = 0.0

        if book_id in top_similar_items_dict:
            similar_books = top_similar_items_dict[book_id]

            for similar_book_id, similarity_score in similar_books.items():
                lookup_key = (user_id, similar_book_id)

                if lookup_key in user_ratings_lookup:
                    rating = user_ratings_lookup[lookup_key]

                    if has_timestamps and current_timestamp is not None:
                        rating_timestamp = timestamp_lookup.get(lookup_key)
                        if rating_timestamp is not None and rating_timestamp >= current_timestamp:
                            continue  # Skip future ratings

                    ratings_list.append(rating)
                    count += 1
                    max_similarity = max(max_similarity, similarity_score)

            if ratings_list:
                score = np.mean(ratings_list)

        std = np.std(ratings_list) if len(ratings_list) > 1 else 0.0

        ibcf_scores.append(score)
        ibcf_counts.append(count)
        ibcf_stds.append(std)
        ibcf_max_sim.append(max_similarity if count > 0 else 0.0)

    # Add features to dataframe
    df[constants.F_IBCF_SCORE] = ibcf_scores
    df[constants.F_IBCF_COUNT] = ibcf_counts
    df[constants.F_IBCF_STD] = ibcf_stds
    df[constants.F_IBCF_MAX_SIM] = ibcf_max_sim

    print(f"IBCF features added:")
    print(f"Coverage: {(~pd.Series(ibcf_scores).isna()).mean():.2%} of rows have IBCF scores")

    return df



def add_easer_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    lambda_reg: float = 500.0,
    n_most_popular: int = 5000,  # Уменьшим до 5000 для скорости
    k_neighbors: int = 50,  # Уменьшим количество соседей
    batch_size: int = 1000,
    use_cache: bool = True,
    cache_dir: str = "./models",
    use_approximate: bool = True  # Использовать приближенное решение для скорости
) -> pd.DataFrame:
    """
    Оптимизированная реализация EASE^R с аппроксимацией для больших данных
    """

    # Константы
    COL_USER_ID = 'user_id'
    COL_BOOK_ID = 'book_id'
    COL_HAS_READ = 'has_read'
    TARGET = 'rating'
    F_EASER_SCORE = 'easer_score'
    F_USER_BOOK_COUNT = 'user_book_count'

    print("=" * 60)
    print("Adding EASE^R features (optimized with approximation)")
    print("=" * 60)

    # Создаем директорию для кеша
    cache_path = Path(cache_dir) / "easer_cache.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Проверяем кеш
    if use_cache and cache_path.exists():
        print(f"Loading cached model from {cache_path}")
        try:
            cache = joblib.load(cache_path)
            W = cache["W"]
            book_id_to_idx = cache["book_id_to_idx"]
            user_ratings_lookup = cache.get("user_ratings_lookup", {})
            print(f"Loaded model: {W.shape[0]} items, sparsity: {W.nnz / (W.shape[0] * W.shape[1]):.6f}")
            print(f"User ratings lookup: {len(user_ratings_lookup)} users")

            # Переходим сразу к вычислению предсказаний
            compute_predictions = True

        except Exception as e:
            print(f"Cache error: {e}, training new model...")
            compute_predictions = False
    else:
        compute_predictions = False

    if not compute_predictions:
        print("Preparing training data...")

        # 1. Фильтруем только прочитанные книги
        rated_train = train_df[train_df[COL_HAS_READ] == 1].copy()
        print(f"Training data: {len(rated_train)} interactions")

        # 2. Ограничиваем по популярным книгам
        if n_most_popular > 0:
            book_counts = rated_train[COL_BOOK_ID].value_counts()
            popular_books = book_counts.head(n_most_popular).index.tolist()
            rated_train = rated_train[rated_train[COL_BOOK_ID].isin(popular_books)]
            print(f"Using {len(popular_books)} most popular books")

        # 3. Создаем категориальные индексы
        print("Creating categorical indices...")
        user_ids = rated_train[COL_USER_ID].astype('category')
        book_ids = rated_train[COL_BOOK_ID].astype('category')

        n_users = len(user_ids.cat.categories)
        n_items = len(book_ids.cat.categories)
        print(f"Users: {n_users}, Items: {n_items}")

        # 4. Создаем sparse user-item матрицу
        print("Creating sparse user-item matrix...")
        X = sparse.coo_matrix(
            (rated_train[TARGET].values.astype(np.float32),
             (user_ids.cat.codes, book_ids.cat.codes)),
            shape=(n_users, n_items)
        ).tocsr()

        density = X.nnz / (n_users * n_items)
        print(f"Sparse matrix: {X.shape[0]} users x {X.shape[1]} items, "
              f"density: {density:.6f}, nnz: {X.nnz}")

        # 5. Обучаем модель EASE^R (оптимизированная версия)
        print(f"\nTraining EASE^R on {n_items} items...")

        # Метод 1: Аппроксимация через псевдообратную матрицу (быстрее)
        if use_approximate or n_items > 3000:
            print("Using approximate solution (pseudoinverse)...")

            # Вычисляем G = X^T X
            print("Step 1: Computing G = X^T X...")
            G = X.T @ X

            # Добавляем регуляризацию на диагональ
            print("Step 2: Adding regularization...")
            G_reg = G.copy()
            diag_indices = np.arange(n_items)
            G_reg[diag_indices, diag_indices] += lambda_reg

            # Используем приближенную инверсию через SVD усечение
            print("Step 3: Computing approximate inverse via SVD...")
            try:
                # Пробуем вычислить обратную напрямую для sparse
                if n_items <= 5000:
                    # Для относительно небольших матриц
                    G_dense = G_reg.toarray()
                    W_dense = np.linalg.inv(G_dense)
                    np.fill_diagonal(W_dense, 0)
                    W = sparse.csr_matrix(W_dense)
                else:
                    # Для больших матриц используем приближение
                    from scipy.sparse.linalg import svds

                    # Вычисляем топ-k сингулярных векторов
                    k = min(100, n_items - 1)
                    print(f"Computing top-{k} SVD components...")
                    U, s, Vt = svds(G_reg, k=k)

                    # Приближенная обратная: V * S^-1 * U^T
                    s_inv = 1.0 / s
                    W_approx = (Vt.T * s_inv) @ U.T

                    # Обнуляем диагональ и обрезаем малые значения
                    np.fill_diagonal(W_approx, 0)
                    W_approx[np.abs(W_approx) < 1e-6] = 0
                    W = sparse.csr_matrix(W_approx)

            except Exception as e:
                print(f"SVD failed: {e}, using cosine similarity fallback...")
                # Fallback: используем косинусную схожесть как аппроксимацию
                item_norms = np.sqrt(np.array(G.diagonal()).flatten())
                item_norms[item_norms == 0] = 1e-10

                W_data = []
                W_indices = []
                W_indptr = [0]

                for i in tqdm(range(n_items), desc="Computing cosine similarities"):
                    if item_norms[i] > 0:
                        # Получаем ненулевые схожести
                        row_start = G.indptr[i]
                        row_end = G.indptr[i + 1]

                        if row_start < row_end:
                            indices = G.indices[row_start:row_end]
                            values = G.data[row_start:row_end]

                            # Вычисляем косинусные схожести
                            for idx, val in zip(indices, values):
                                if i != idx and item_norms[idx] > 0:
                                    similarity = val / (item_norms[i] * item_norms[idx])

                                    # Оставляем только топ-k соседей
                                    if len(W_indices) - W_indptr[i] < k_neighbors:
                                        W_indices.append(idx)
                                        W_data.append(similarity)
                                    else:
                                        # Находим минимальный вес среди сохраненных
                                        min_idx = np.argmin(np.abs(np.array(W_data[W_indptr[i]:])))
                                        if abs(similarity) > abs(W_data[W_indptr[i] + min_idx]):
                                            W_indices[W_indptr[i] + min_idx] = idx
                                            W_data[W_indptr[i] + min_idx] = similarity

                    W_indptr.append(len(W_indices))

                W = sparse.csr_matrix(
                    (W_data, W_indices, W_indptr),
                    shape=(n_items, n_items),
                    dtype=np.float32
                )

        else:
            # Метод 2: Точное решение (только для небольших данных)
            print("Using exact solution...")
            G = X.T @ X
            G_reg = G.copy()
            diag_indices = np.arange(n_items)
            G_reg[diag_indices, diag_indices] += lambda_reg

            # Вычисляем обратную матрицу
            print("Computing matrix inverse...")
            try:
                W_dense = np.linalg.inv(G_reg.toarray())
                np.fill_diagonal(W_dense, 0)
                W = sparse.csr_matrix(W_dense)
            except:
                print("Inverse failed, using pseudoinverse...")
                W_dense = np.linalg.pinv(G_reg.toarray())
                np.fill_diagonal(W_dense, 0)
                W = sparse.csr_matrix(W_dense)

        print(f"Weight matrix: {W.shape}, nnz: {W.nnz}, "
              f"sparsity: {W.nnz / (n_items * n_items):.6f}")

        # 6. Создаем маппинги
        book_id_to_idx = {book_id: idx for idx, book_id in enumerate(book_ids.cat.categories)}

        # 7. Создаем lookup таблицу user ratings (оптимизированная версия)
        print("\nBuilding optimized user ratings lookup...")

        # Сначала создаем dense матрицу пользовательских рейтингов
        user_ratings_lookup = {}

        # Используем groupby и agg для скорости
        user_book_ratings = rated_train.groupby([COL_USER_ID, COL_BOOK_ID])[TARGET].first().reset_index()

        # Создаем маппинг user_id -> индекс
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids.cat.categories)}

        for user_id in tqdm(user_ids.cat.categories, desc="Creating user vectors"):
            # Фильтруем рейтинги для конкретного пользователя
            user_ratings = user_book_ratings[user_book_ratings[COL_USER_ID] == user_id]

            if not user_ratings.empty:
                # Создаем sparse вектор
                col_indices = []
                values = []

                for _, row in user_ratings.iterrows():
                    book_id = row[COL_BOOK_ID]
                    if book_id in book_id_to_idx:
                        col_indices.append(book_id_to_idx[book_id])
                        values.append(row[TARGET])

                if col_indices:
                    user_vec = sparse.csr_matrix(
                        (values, ([0] * len(col_indices), col_indices)),
                        shape=(1, n_items),
                        dtype=np.float32
                    )
                    user_ratings_lookup[user_id] = user_vec

        print(f"User ratings lookup created: {len(user_ratings_lookup)} users")

        # 8. Сохраняем в кеш
        print(f"\nSaving model to cache: {cache_path}")
        try:
            joblib.dump(
                {
                    "W": W,
                    "book_id_to_idx": book_id_to_idx,
                    "user_ratings_lookup": user_ratings_lookup,
                    "n_items": n_items
                },
                cache_path,
                compress=3
            )
            print("Model saved successfully")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    # Вычисляем предсказания для df
    print(f"\nComputing EASE^R predictions for {len(df)} rows...")

    # Создаем массив для результатов
    easer_scores = np.full(len(df), np.nan, dtype=np.float32)

    # Подсчитываем статистику покрытия
    total_found = 0

    # Обрабатываем построчно (batch processing может быть медленнее для sparse операций)
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Scoring")):
        user_id = row[COL_USER_ID]
        book_id = row[COL_BOOK_ID]

        # Проверяем наличие данных
        if (user_id in user_ratings_lookup and
            book_id in book_id_to_idx):

            user_vec = user_ratings_lookup[user_id]
            target_idx = book_id_to_idx[book_id]

            # Вычисляем score
            if user_vec.nnz > 0:
                try:
                    # Умножаем только ненулевые элементы для скорости
                    score = 0.0

                    # Получаем ненулевые индексы и значения пользовательского вектора
                    user_indices = user_vec.indices
                    user_values = user_vec.data

                    # Получаем соответствующий столбец W
                    W_col_start = W.indptr[target_idx]
                    W_col_end = W.indptr[target_idx + 1]

                    if W_col_start < W_col_end:
                        W_col_indices = W.indices[W_col_start:W_col_end]
                        W_col_values = W.data[W_col_start:W_col_end]

                        # Создаем lookup для быстрого доступа к весам W
                        W_dict = dict(zip(W_col_indices, W_col_values))

                        # Вычисляем скалярное произведение
                        for user_idx, user_val in zip(user_indices, user_values):
                            if user_idx in W_dict:
                                score += user_val * W_dict[user_idx]

                    easer_scores[idx] = score
                    total_found += 1

                except Exception as e:
                    # В случае ошибки оставляем NaN
                    pass

    # Добавляем фичи в DataFrame
    df = df.copy()
    df[F_EASER_SCORE] = easer_scores

    # Добавляем количество прочитанных книг пользователя
    print("Adding user book count feature...")
    user_book_counts = train_df[train_df[COL_HAS_READ] == 1].groupby(
        COL_USER_ID
    )[COL_BOOK_ID].nunique()

    df[F_USER_BOOK_COUNT] = df[COL_USER_ID].map(user_book_counts).fillna(0)

    # Статистика
    coverage = total_found / len(df)
    print(f"\nEASE^R Statistics:")
    print(f"  Coverage:        {coverage:.2%} ({total_found}/{len(df)})")

    if total_found > 0:
        valid_scores = df[F_EASER_SCORE].dropna()
        print(f"  Min score:       {valid_scores.min():.4f}")
        print(f"  Max score:       {valid_scores.max():.4f}")
        print(f"  Mean score:      {valid_scores.mean():.4f}")
        print(f"  Std score:       {valid_scores.std():.4f}")

    print(f"  User book count: min={df[F_USER_BOOK_COUNT].min()}, "
          f"max={df[F_USER_BOOK_COUNT].max()}, "
          f"mean={df[F_USER_BOOK_COUNT].mean():.1f}")

    # Заполняем пропуски
    nan_count = df[F_EASER_SCORE].isna().sum()
    if nan_count > 0:
        median_score = df[F_EASER_SCORE].median()
        df[F_EASER_SCORE] = df[F_EASER_SCORE].fillna(median_score)
        print(f"Filled {nan_count} NaN values with median: {median_score:.4f}")

    # Нормализуем scores для лучшей стабильности
    if total_found > 1:
        score_mean = df[F_EASER_SCORE].mean()
        score_std = df[F_EASER_SCORE].std()
        if score_std > 0:
            df[F_EASER_SCORE] = (df[F_EASER_SCORE] - score_mean) / score_std
            print(f"Normalized scores: mean=0, std=1")

    print(f"\n✓ EASE^R features added successfully!")
    print("=" * 60)

    return df

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
    #df = add_text_features(df, train_df, descriptions_df)
    #df = add_ibcf_features(df, train_df)
    #df = add_ubcf_features(df, train_df)
    df = add_easer_features(df, train_df)
    df = add_nomic_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for CatBoost
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
