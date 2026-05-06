"""V2 prompt safety classifier using a local transformer-based model."""

from typing import Tuple


class PromptSafetyClassifierV2:
    """A local ML-based classifier for prompt safety detection.

    Uses a transformer model to classify prompts into three categories:
    - "safe": the prompt is benign
    - "suspicious": the prompt may require review
    - "unsafe": the prompt is clearly harmful or violates safety guidelines

    Args:
        model_name (str): HuggingFace model identifier for the safety classifier.
            Defaults to "lmsys/toxicity-llm-judge" compatible pipeline.
        unsafe_threshold (float): Score above which a prompt is classified as
            "unsafe". Defaults to 0.8.
        suspicious_threshold (float): Score above which (but below
            ``unsafe_threshold``) a prompt is classified as "suspicious".
            Defaults to 0.5.
    """

    DEFAULT_MODEL = "Marqo/dunzhang-stella_en_400M_v5-prompt-safety"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        unsafe_threshold: float = 0.8,
        suspicious_threshold: float = 0.5,
    ):
        if unsafe_threshold <= suspicious_threshold:
            raise ValueError(
                "unsafe_threshold must be greater than suspicious_threshold"
            )
        self.model_name = model_name
        self.unsafe_threshold = unsafe_threshold
        self.suspicious_threshold = suspicious_threshold
        self._pipeline = None

    def _load_pipeline(self):
        """Lazily load the classification pipeline."""
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "The 'transformers' package is required to use PromptSafetyClassifierV2. "
                "Install it with: pip install transformers"
            ) from e

        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            truncation=True,
            max_length=512,
        )

    def _get_unsafe_score(self, model_output: dict) -> float:
        """Extract an unsafe probability score from the model output.

        The pipeline returns a list of dicts with 'label' and 'score' keys.
        If the top label indicates unsafe content, its score is returned directly.
        Otherwise, 1 - score is returned so the result is always the
        probability of the prompt being unsafe.
        """
        label: str = model_output["label"].lower()
        score: float = model_output["score"]
        # Labels that indicate unsafe content
        unsafe_labels = {"unsafe", "toxic", "label_1", "negative", "1"}
        if label in unsafe_labels or any(u in label for u in ("unsafe", "toxic")):
            return score
        # Safe/benign label — invert the score
        return 1.0 - score

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify a prompt and return a category and confidence score.

        Args:
            text (str): The prompt text to classify.

        Returns:
            Tuple[str, float]: A tuple of (category, score) where category is
                one of "safe", "suspicious", or "unsafe", and score is the
                model's confidence that the prompt is unsafe (0.0 – 1.0).
        """
        if self._pipeline is None:
            self._load_pipeline()

        result = self._pipeline(text)
        # pipeline returns a list; take the first (top) prediction
        if isinstance(result, list):
            result = result[0]

        score = self._get_unsafe_score(result)

        if score >= self.unsafe_threshold:
            category = "unsafe"
        elif score >= self.suspicious_threshold:
            category = "suspicious"
        else:
            category = "safe"

        return category, score
