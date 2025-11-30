"""
Configuration file for the NTO ML competition baseline.
"""

from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants

# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"


# --- PARAMETERS ---
N_SPLITS = 5  # Deprecated: kept for backwards compatibility, not used in temporal split
RANDOM_STATE = 42
TARGET = constants.COL_TARGET  # Alias for consistency
# --- TEMPORAL SPLIT CONFIG ---
# Ratio of data to use for training (0 < TEMPORAL_SPLIT_RATIO < 1)
# 0.8 means 80% of data points (by timestamp) go to train, 20% to validation
TEMPORAL_SPLIT_RATIO = 0.8

# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 50
MODEL_FILENAME_PATTERN = "catboost_fold_{fold}.cbm"  # Deprecated: kept for backwards compatibility
MODEL_FILENAME = "catboost_model.cbm"  # Single model filename for temporal split

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# --- NOMIC PARAMETERS ---
NOMIC_MODEL_NAME = constants.NOMIC_MODEL_NAME
NOMIC_BATCH_SIZE = 8
NOMIC_MAX_LENGTH = 512
NOMIC_EMBEDDING_DIM = 768
NOMIC_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
NOMIC_GPU_MEMORY_FRACTION = 0.75


# --- FEATURES ---
# Categorical features for CatBoost (excluding numeric features like age and publication_year)
CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AUTHOR_ID,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]

# --- MODEL PARAMETERS ---
CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'custom_metric': ['RMSE', 'MAE'],
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 50,
    "max_leaves": 32,
    "min_data_in_leaf": 300,
    "subsample": 0.7,
    "rsm": 0.8,
    "bootstrap_type": "Bernoulli",
    "grow_policy": "Lossguide",
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "thread_count": -1,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS
}
#
