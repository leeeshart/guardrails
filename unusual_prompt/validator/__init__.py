"""
unusual_prompt.validator package.

Imports of the Guardrails-based validators are deferred so that helper
modules (intent_patterns, model_config) can be imported without a full
guardrails installation (e.g. in the model training script).
"""


def __getattr__(name: str):
    if name == "UnusualPrompt":
        from .main import UnusualPrompt  # noqa: PLC0415

        return UnusualPrompt
    if name == "UnusualPromptV2":
        from .v2_classifier import UnusualPromptV2  # noqa: PLC0415

        return UnusualPromptV2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["UnusualPrompt", "UnusualPromptV2"]
