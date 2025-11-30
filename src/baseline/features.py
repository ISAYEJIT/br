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
    n_similar_users: int = 30,
    use_timestamp_filtering: bool = True
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
    n_similar_items: int = 50
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

    # Step 2: Prepare user rating lookup
    print("Preparing user rating lookup...")
    rated_train = train_df[train_df[constants.COL_HAS_READ] == 1].copy()

    # Create lookup: (user_id, book_id) -> rating
    user_ratings_lookup = {}
    for _, row in rated_train.iterrows():
        key = (row[constants.COL_USER_ID], row[constants.COL_BOOK_ID])
        user_ratings_lookup[key] = row[config.TARGET]

    # Step 3: Compute IBCF features
    print(f"Computing IBCF features for {len(df)} rows...")

    ibcf_scores = []
    ibcf_counts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="IBCF features"):
        user_id = row[constants.COL_USER_ID]
        book_id = row[constants.COL_BOOK_ID]

        score = np.nan
        count = 0

        # Check if we have similar books for this book
        if book_id in top_similar_items_dict:
            similar_books = top_similar_items_dict[book_id]

            ratings_list = []

            # Look for similar books this user has rated
            for similar_book_id in similar_books.keys():
                lookup_key = (user_id, similar_book_id)

                if lookup_key in user_ratings_lookup:
                    rating = user_ratings_lookup[lookup_key]
                    ratings_list.append(rating)
                    count += 1

            if ratings_list:
                score = np.mean(ratings_list)

        ibcf_scores.append(score)
        ibcf_counts.append(count)

    df[constants.F_IBCF_SCORE] = ibcf_scores
    df[constants.F_IBCF_COUNT] = ibcf_counts

    print(f"IBCF features added:")
    print(f"  - {constants.F_IBCF_SCORE}: avg rating from similar books")
    print(f"  - {constants.F_IBCF_COUNT}: number of similar books user rated")
    print(f"Coverage: {(~pd.Series(ibcf_scores).isna()).mean():.2%} of rows have IBCF scores")

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


def add_bert_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds BERT embeddings from book descriptions.

    Extracts 768-dimensional embeddings using a pre-trained Russian BERT model.
    Embeddings are cached on disk to avoid recomputation on subsequent runs.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        _train_df (pd.DataFrame): The training portion (for consistency, not used for BERT).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with BERT embeddings added.
    """
    print("Adding text features (BERT embeddings)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME

    # Check if embeddings are already cached
    if embeddings_path.exists():
        print(f"Loading cached BERT embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing BERT embeddings (this may take a while)...")
        print(f"Using device: {config.BERT_DEVICE}")

        # Limit GPU memory usage to prevent OOM errors
        if config.BERT_DEVICE == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(config.BERT_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.BERT_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        model.to(config.BERT_DEVICE)
        model.eval()

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
        num_batches = (len(descriptions) + config.BERT_BATCH_SIZE - 1) // config.BERT_BATCH_SIZE

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing BERT batches", unit="batch"):
                start_idx = batch_idx * config.BERT_BATCH_SIZE
                end_idx = min(start_idx + config.BERT_BATCH_SIZE, len(descriptions))
                batch_descriptions = descriptions[start_idx:end_idx]
                batch_book_ids = book_ids[start_idx:end_idx]

                # Tokenize batch
                encoded = tokenizer(
                    batch_descriptions,
                    padding=True,
                    truncation=True,
                    max_length=config.BERT_MAX_LENGTH,
                    return_tensors="pt",
                )

                # Move to device
                encoded = {k: v.to(config.BERT_DEVICE) for k, v in encoded.items()}

                # Get model outputs
                outputs = model(**encoded)

                # Mean pooling: average over sequence length dimension
                # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
                attention_mask = encoded["attention_mask"]
                # Expand attention mask to match hidden_size dimension for broadcasting
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()

                # Sum embeddings, weighted by attention mask
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
                # Sum attention mask values for normalization
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

                # Mean pooling
                mean_pooled = sum_embeddings / sum_mask

                # Convert to numpy and store
                batch_embeddings = mean_pooled.cpu().numpy()

                for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                    embeddings_dict[book_id] = embedding

                # Small pause between batches to let GPU cool down and prevent overheating
                if config.BERT_DEVICE == "cuda":
                    time.sleep(0.2)  # 200ms pause between batches

        # Save embeddings for future use
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved BERT embeddings to {embeddings_path}")

    # Map embeddings to DataFrame rows by book_id
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()

    # Create embedding matrix
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            # Zero embedding for books without descriptions
            embeddings_list.append(np.zeros(config.BERT_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)

    # Apply PCA to reduce dimensionality and prevent overfitting
    pca_path = config.MODEL_DIR / constants.BERT_PCA_FILENAME

    if pca_path.exists():
        print(f"Loading PCA model from {pca_path}")
        pca = joblib.load(pca_path)
        embeddings_reduced = pca.transform(embeddings_array)
        print(f"Applied PCA: {config.BERT_EMBEDDING_DIM} -> {config.BERT_PCA_COMPONENTS} dimensions")
    else:
        print(f"Fitting PCA to reduce BERT embeddings from {config.BERT_EMBEDDING_DIM} to {config.BERT_PCA_COMPONENTS} dimensions...")
        pca = PCA(n_components=config.BERT_PCA_COMPONENTS, random_state=config.RANDOM_STATE)
        embeddings_reduced = pca.fit_transform(embeddings_array)

        # Save PCA model
        joblib.dump(pca, pca_path)
        print(f"Saved PCA model to {pca_path}")
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    # Create DataFrame with reduced BERT features
    bert_feature_names = [f"bert_{i}" for i in range(config.BERT_PCA_COMPONENTS)]
    bert_df = pd.DataFrame(embeddings_reduced, columns=bert_feature_names, index=df.index)

    # Concatenate BERT features with main DataFrame
    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(bert_feature_names)} BERT features (reduced from {config.BERT_EMBEDDING_DIM} via PCA).")
    return df_with_bert


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values using a defined strategy.

    Args:
        df: The DataFrame with missing values
        train_df: The training data, used for calculating fill metrics

    Returns:
        DataFrame with missing values handled
    """
    print("Handling missing values...")

    # Calculate global mean from training data
    global_mean = train_df[config.TARGET].mean()

    # Fill age with median
    if constants.COL_AGE in df.columns:
        age_median = df[constants.COL_AGE].median()
        df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill aggregate features (if they exist)
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

    # Fill avg_rating from book metadata
    if constants.COL_AVG_RATING in df.columns:
        df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts
    if constants.F_BOOK_GENRES_COUNT in df.columns:
        df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # NEW: Fill UBCF features
    if constants.F_UBCF_SCORE in df.columns:
        # Don't fill with global_mean - let the model learn that NaN means "no data"
        # We'll fill with a special value that indicates cold start
        df[constants.F_UBCF_SCORE] = df[constants.F_UBCF_SCORE].fillna(global_mean)
    if constants.F_UBCF_COUNT in df.columns:
        df[constants.F_UBCF_COUNT] = df[constants.F_UBCF_COUNT].fillna(0)
    if constants.F_UBCF_STD in df.columns:
        df[constants.F_UBCF_STD] = df[constants.F_UBCF_STD].fillna(0)
    if constants.F_UBCF_MAX_SIM in df.columns:
        df[constants.F_UBCF_MAX_SIM] = df[constants.F_UBCF_MAX_SIM].fillna(0)

    # NEW: Fill IBCF features
    if constants.F_IBCF_SCORE in df.columns:
        df[constants.F_IBCF_SCORE] = df[constants.F_IBCF_SCORE].fillna(global_mean)
    if constants.F_IBCF_COUNT in df.columns:
        df[constants.F_IBCF_COUNT] = df[constants.F_IBCF_COUNT].fillna(0)

    # Fill TF-IDF features (if they exist)
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    # Fill BERT features
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    # Fill categorical features
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    print("Missing values handled")
    return df


def create_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    include_aggregates: bool = False,
    use_ubcf: bool = True,
    use_ibcf: bool = True,
    use_tfidf: bool = False,
    use_bert: bool = True
) -> pd.DataFrame:
    """Runs the full feature engineering pipeline.

    Args:
        df: The merged DataFrame from data_processing
        book_genres_df: DataFrame mapping books to genres
        descriptions_df: DataFrame with book descriptions
        include_aggregates: If True, compute aggregate features
        use_ubcf: If True, add UBCF features (recommended)
        use_ibcf: If True, add IBCF features (recommended)
        use_tfidf: If True, add TF-IDF features (may cause overfitting)
        use_bert: If True, add BERT features (recommended)

    Returns:
        Final DataFrame with all features engineered
    """
    print("Starting feature engineering pipeline...")
    print(f"Configuration: aggregates={include_aggregates}, ubcf={use_ubcf}, "
          f"ibcf={use_ibcf}, tfidf={use_tfidf}, bert={use_bert}")

    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Aggregate features (if enabled)
    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    # Genre features
    df = add_genre_features(df, book_genres_df)

    # Collaborative filtering features
    if use_ubcf:
        df = add_ubcf_features(
            df,
            train_df,
            n_similar_users=30,  # Can tune this
            use_timestamp_filtering=True  # Prevent leakage
        )

    if use_ibcf:
        df = add_ibcf_features(
            df,
            train_df,
            n_similar_items=50  # Can tune this
        )

    if use_tfidf:
        print("WARNING: TF-IDF may cause overfitting. Consider using BERT only.")
        df = add_text_features(df, train_df, descriptions_df)

    if use_bert:
        df = add_bert_features(df, train_df, descriptions_df)

    df = handle_missing_values(df, train_df)

    # Ensure categorical features
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    print(f"Final shape: {df.shape}")

    return df
