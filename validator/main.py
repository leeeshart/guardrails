"""UnusualPrompt validator – detects unusual or potentially unsafe prompts."""

import json
from typing import Any, Callable, Dict, Optional, Union

import litellm
from guardrails.validator_base import Validator, register_validator
from guardrails_ai.types import FailResult, PassResult, ValidationResult

from .v2_classifier import PromptSafetyClassifierV2

UNUSUAL_PROMPT_SYSTEM_MESSAGE = (
    "You are a prompt safety classifier. "
    "Your task is to evaluate whether a user prompt is unusual, harmful, or "
    "potentially unsafe. "
    "Reply with a JSON object containing:\n"
    '  "safe": true if the prompt is benign, false otherwise\n'
    '  "reason": a brief explanation of your decision\n'
    "Be conservative — when in doubt, flag the prompt as unsafe."
)

UNUSUAL_PROMPT_USER_TEMPLATE = (
    "Is the following user prompt unusual or potentially unsafe?\n\n"
    "Prompt: {value}\n\n"
    'Respond ONLY with valid JSON: {{"safe": true/false, "reason": "..."}}'
)


@register_validator(name="guardrails/unusual_prompt", data_type="string")
class UnusualPrompt(Validator):
    """Validates that a user prompt is not unusual or potentially unsafe.

    Uses either an LLM call (v1, default) or a local ML classifier (v2) to
    determine whether a prompt is safe, suspicious, or unsafe.

    **Key Properties**

    | Property                      | Description                                   |
    | ----------------------------- | --------------------------------------------- |
    | Name for `format` attribute   | `guardrails/unusual_prompt`                   |
    | Supported data types          | `string`                                      |
    | Programmatic fix              | None                                          |

    Args:
        llm_callable (str): The LiteLLM model string used for v1 LLM-based
            validation. Defaults to ``"gpt-3.5-turbo"``.
        use_v2_classifier (bool): When ``True``, use the local
            :class:`~validator.v2_classifier.PromptSafetyClassifierV2` instead
            of calling an LLM. Defaults to ``False``.
        v2_model_name (str, optional): HuggingFace model identifier passed to
            :class:`~validator.v2_classifier.PromptSafetyClassifierV2`.
            Only used when ``use_v2_classifier=True``.
        v2_unsafe_threshold (float): Score threshold above which a prompt is
            classified as "unsafe" by the v2 classifier. Defaults to ``0.8``.
        v2_suspicious_threshold (float): Score threshold above which (but below
            ``v2_unsafe_threshold``) a prompt is classified as "suspicious" by
            the v2 classifier. Defaults to ``0.5``.
        on_fail (Callable, optional): The action to take when validation fails.
    """

    def __init__(
        self,
        llm_callable: str = "gpt-3.5-turbo",
        use_v2_classifier: bool = False,
        v2_model_name: Optional[str] = None,
        v2_unsafe_threshold: float = 0.8,
        v2_suspicious_threshold: float = 0.5,
        on_fail: Optional[Union[Callable[..., Any], str]] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail,
            llm_callable=llm_callable,
            use_v2_classifier=use_v2_classifier,
            **kwargs,
        )
        self.llm_callable = llm_callable
        self.use_v2_classifier = use_v2_classifier

        if use_v2_classifier:
            classifier_kwargs: Dict[str, Any] = {
                "unsafe_threshold": v2_unsafe_threshold,
                "suspicious_threshold": v2_suspicious_threshold,
            }
            if v2_model_name is not None:
                classifier_kwargs["model_name"] = v2_model_name
            self.v2_classifier = PromptSafetyClassifierV2(**classifier_kwargs)

    # ------------------------------------------------------------------
    # Public validation entry-point
    # ------------------------------------------------------------------

    def validate(self, value: Any, metadata: Dict[str, Any] = {}) -> ValidationResult:
        if self.use_v2_classifier:
            return self._validate_v2(value)
        return self._validate_llm(value)

    # ------------------------------------------------------------------
    # V2 – local ML classifier
    # ------------------------------------------------------------------

    def _validate_v2(self, value: str) -> ValidationResult:
        """Validate using the local PromptSafetyClassifierV2."""
        category, score = self.v2_classifier.classify(value)

        if category == "unsafe":
            return FailResult(
                error_message=(
                    f"Prompt detected as unsafe (score: {score:.2f})"
                )
            )
        if category == "suspicious":
            return FailResult(
                error_message=(
                    f"Prompt is suspicious and may need review (score: {score:.2f})"
                )
            )
        return PassResult()

    # ------------------------------------------------------------------
    # V1 – LLM-based validation
    # ------------------------------------------------------------------

    def _validate_llm(self, value: str) -> ValidationResult:
        """Validate by asking an LLM whether the prompt is unusual/unsafe."""
        messages = [
            {"role": "system", "content": UNUSUAL_PROMPT_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": UNUSUAL_PROMPT_USER_TEMPLATE.format(value=value),
            },
        ]

        response = litellm.completion(
            model=self.llm_callable,
            messages=messages,
        )

        raw_content: str = response.choices[0].message.content or ""

        try:
            result = json.loads(raw_content)
            is_safe: bool = bool(result.get("safe", True))
            reason: str = result.get("reason", "")
        except (json.JSONDecodeError, AttributeError):
            # If we can't parse the response, treat it as safe to avoid
            # false positives caused by LLM formatting issues.
            is_safe = True
            reason = ""

        if not is_safe:
            return FailResult(
                error_message=(
                    f"Prompt flagged as unusual or unsafe by LLM classifier"
                    + (f": {reason}" if reason else "")
                )
            )
        return PassResult()
