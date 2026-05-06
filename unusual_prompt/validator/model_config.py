"""
Model configuration for the unusual_prompt v2 ML classifier.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _PACKAGE_DIR / "models"

MODEL_PATH: Path = MODELS_DIR / "model_v2.pkl"
VECTORIZER_PATH: Path = MODELS_DIR / "vectorizer_v2.pkl"

# ---------------------------------------------------------------------------
# Vectorizer hyperparameters (TF-IDF)
# ---------------------------------------------------------------------------

TFIDF_CONFIG: dict = {
    "analyzer": "char_wb",
    "ngram_range": (2, 4),
    "max_features": 50_000,
    "sublinear_tf": True,
    "min_df": 1,
    "strip_accents": "unicode",
}

# ---------------------------------------------------------------------------
# Classifier hyperparameters (LogisticRegression)
# ---------------------------------------------------------------------------

CLASSIFIER_CONFIG: dict = {
    "C": 5.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------

# Probability threshold above which a prompt is flagged as unusual.
# Lowering this increases recall (catches more attacks) at the cost of
# more false positives.
DECISION_THRESHOLD: float = float(os.getenv("UNUSUAL_PROMPT_THRESHOLD", "0.5"))

# Label used by the classifier for the "unusual / adversarial" class.
ADVERSARIAL_LABEL: int = 1
BENIGN_LABEL: int = 0

# Human-readable class names (used in error messages).
CLASS_NAMES: dict = {
    BENIGN_LABEL: "benign",
    ADVERSARIAL_LABEL: "adversarial",
}

# ---------------------------------------------------------------------------
# Training settings (used by the model training script)
# ---------------------------------------------------------------------------

TRAINING_CONFIG: dict = {
    "test_size": 0.2,
    "random_state": 42,
    "cross_val_folds": 5,
}
