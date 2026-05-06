"""
v2 offline ML classifier for unusual / adversarial prompts.

This module provides :class:`UnusualPromptV2`, a Guardrails validator that
combines a TF-IDF + Logistic-Regression model with fast regex intent-pattern
flags to detect adversarial prompts **without** any LLM API calls.

Architecture
------------
1. **Intent patterns** (:mod:`.intent_patterns`) — O(1) regex checks that
   catch well-known attack signatures immediately.
2. **ML classifier** — a character-level TF-IDF + Logistic Regression model
   loaded from ``models/model_v2.pkl`` and ``models/vectorizer_v2.pkl``.
   The pattern flags are appended as extra binary features before
   classification, giving the model explicit signal about structural attacks.

The two signals are combined with a soft *or*: if either the pattern score or
the ML probability exceeds the configured threshold the prompt is flagged.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

from .intent_patterns import INTENT_PATTERNS, flag_patterns
from .model_config import (
    ADVERSARIAL_LABEL,
    CLASS_NAMES,
    DECISION_THRESHOLD,
    MODEL_PATH,
    VECTORIZER_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading (lazy, singleton)
# ---------------------------------------------------------------------------

_model = None
_vectorizer = None


def _load_model() -> Tuple[Any, Any]:
    """Load the pickled vectorizer and classifier (lazy singleton).

    Returns:
        ``(vectorizer, classifier)`` sklearn objects.

    Raises:
        FileNotFoundError: If either pickle file is missing.
    """
    global _model, _vectorizer  # noqa: PLW0603
    if _vectorizer is None or _model is None:
        for path in (VECTORIZER_PATH, MODEL_PATH):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Model artifact not found: {path}. "
                    "Run the training script first:  "
                    "python -m unusual_prompt.scripts.train_model"
                )
        with open(VECTORIZER_PATH, "rb") as fh:
            _vectorizer = pickle.load(fh)  # noqa: S301
        with open(MODEL_PATH, "rb") as fh:
            _model = pickle.load(fh)  # noqa: S301
        logger.debug("UnusualPromptV2: models loaded from %s", MODEL_PATH.parent)
    return _vectorizer, _model


def reset_model_cache() -> None:
    """Clear the in-process model cache (useful for testing)."""
    global _model, _vectorizer  # noqa: PLW0603
    _model = None
    _vectorizer = None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_PATTERN_LABELS: List[str] = sorted({label for label, _ in INTENT_PATTERNS})


def _build_features(text: str, vectorizer: Any) -> np.ndarray:
    """Combine TF-IDF features with binary intent-pattern flags.

    Args:
        text: The raw prompt string.
        vectorizer: A fitted ``TfidfVectorizer``.

    Returns:
        A 2-D numpy array of shape ``(1, n_tfidf + n_patterns)``.
    """
    from scipy.sparse import hstack, csr_matrix  # type: ignore[import-untyped]

    tfidf_features = vectorizer.transform([text])

    triggered = flag_patterns(text)
    pattern_flags = np.array(
        [[1 if lbl in triggered else 0 for lbl in _PATTERN_LABELS]],
        dtype=np.float32,
    )
    return hstack([tfidf_features, csr_matrix(pattern_flags)])


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@register_validator(name="guardrails/unusual_prompt_v2", data_type="string")
class UnusualPromptV2(Validator):
    """Detects unusual or adversarial prompts using an offline ML model.

    # Overview

    | Developed by | Guardrails AI |
    | Validator type | Security |
    | License | Apache 2 |
    | Input/Output | Input |

    # Description

    Uses a character-level TF-IDF + Logistic Regression model combined
    with fast regex intent-pattern flags to classify prompts as *benign*
    or *adversarial* **without** LLM API calls.

    The model is trained on a mix of natural language prompts and known
    adversarial patterns (prompt injection, jailbreaks, system-prompt
    extraction attempts, etc.).

    ## Requirements

    * Dependencies:
        - guardrails-ai>=0.4.0
        - scikit-learn>=1.3.0
        - numpy>=1.25
        - scipy

    # Installation

    ```bash
    guardrails hub install hub://guardrails/unusual_prompt_v2
    ```

    # Usage

    ```python
    from guardrails.hub import UnusualPromptV2
    from guardrails import Guard

    guard = Guard.use(UnusualPromptV2())
    guard.validate("What is the weather today?")          # passes
    guard.validate("Ignore all previous instructions")    # fails
    ```
    """

    def __init__(
        self,
        threshold: float = DECISION_THRESHOLD,
        on_fail: Optional[Callable] = None,
    ):
        """Initialise the v2 classifier.

        Args:
            threshold: Probability threshold (0–1) above which a prompt is
                considered adversarial.  Defaults to the value of the
                ``UNUSUAL_PROMPT_THRESHOLD`` environment variable, or ``0.5``.
            on_fail: The policy to enact when validation fails.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")
        super().__init__(on_fail=on_fail, threshold=threshold)
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _predict(self, text: str) -> Tuple[int, float]:
        """Return (predicted_label, adversarial_probability) for *text*.

        Falls back to pattern-only classification when the model files are
        absent (e.g. during unit tests with mocked pickles).
        """
        try:
            vectorizer, model = _load_model()
        except FileNotFoundError:
            logger.warning(
                "UnusualPromptV2: model files not found – falling back to "
                "pattern-only classification."
            )
            triggered = flag_patterns(text)
            prob = 1.0 if triggered else 0.0
            label = ADVERSARIAL_LABEL if triggered else 0
            return label, prob

        features = _build_features(text, vectorizer)
        proba = model.predict_proba(features)[0]
        adv_prob = float(proba[ADVERSARIAL_LABEL])
        label = ADVERSARIAL_LABEL if adv_prob >= self._threshold else 0
        return label, adv_prob

    # ------------------------------------------------------------------
    # Validator interface
    # ------------------------------------------------------------------

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """Validate that *value* is not an adversarial prompt.

        Args:
            value: The prompt string to validate.
            metadata: Unused; present for interface compatibility.

        Returns:
            :class:`~guardrails.validator_base.PassResult` when benign, or
            :class:`~guardrails.validator_base.FailResult` with an explanation
            when adversarial.
        """
        text = str(value)
        label, adv_prob = self._predict(text)

        if label == ADVERSARIAL_LABEL:
            triggered = flag_patterns(text)
            pattern_detail = (
                f" Triggered patterns: {', '.join(triggered)}." if triggered else ""
            )
            return FailResult(
                error_message=(
                    f"Prompt classified as {CLASS_NAMES[ADVERSARIAL_LABEL]} "
                    f"(confidence {adv_prob:.0%}).{pattern_detail}"
                ),
            )
        return PassResult()
