"""
Configuration file for the NTO ML competition baseline.
"""

from pathlib import Path
import numpy as np

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
MODEL_FILENAME_PATTERN = "catboost_fold_{fold}.cbm"
MODEL_FILENAME = "catboost_model.cbm"

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# --- BERT PARAMETERS ---
BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 8
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768
BERT_PCA_COMPONENTS = 50  # Уменьшаем размерность BERT эмбеддингов через PCA для уменьшения переобучения
BERT_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
# Limit GPU memory usage to 50% to prevent overheating and OOM errors
BERT_GPU_MEMORY_FRACTION = 0.75


# --- FEATURES ---
# ВАЖНО: user_id и book_id НЕ должны быть категориальными признаками!
# Они вызывают сильное переобучение, так как модель запоминает конкретные ID.
# Используем только агрегатные признаки (user_mean_rating, book_mean_rating).
CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]


class ServerMetric:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        y_true = np.array(target)
        y_pred = np.array(approxes[0])

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        score = 1 - (0.5 * rmse / 10 + 0.5 * mae / 10)

        return score, 1


# --- MODEL PARAMETERS ---

CATBOOST_PARAMS = {
    'loss_function': 'MAE',
    'eval_metric': ServerMetric(),
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
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}

CATBOOST_FIT_PARAMS = {
    "use_best_model": True,
}
