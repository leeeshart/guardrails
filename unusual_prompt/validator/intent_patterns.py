"""
Intent pattern flags for detecting unusual or adversarial prompts.

These patterns cover common prompt injection, jailbreak, and
system-prompt-extraction techniques observed in the wild.
"""

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Each entry is (label, compiled_regex).
# The label is a short tag that explains the category of the match.
# ---------------------------------------------------------------------------

INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # --- Instruction override ---
    (
        "instruction_override",
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?|constraints?)",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override",
        re.compile(
            r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override",
        re.compile(
            r"forget\s+(everything|all)\s+(you('ve)?\s+)?(been\s+)?(told|learned|know|said)",
            re.IGNORECASE,
        ),
    ),
    # --- Role / persona hijacking ---
    (
        "persona_hijack",
        re.compile(
            r"\byou\s+are\s+(now\s+|no\s+longer\s+|a\s+new\s+|an?\s+)?\b(DAN|JAILBREAK|evil\s+AI|unrestricted|uncensored)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "persona_hijack",
        re.compile(
            r"pretend\s+(you\s+are|to\s+be|you('re)?)\s+.{0,60}(no\s+restrictions?|no\s+limits?|uncensored|without\s+guidelines?)",
            re.IGNORECASE,
        ),
    ),
    (
        "persona_hijack",
        re.compile(
            r"\bact\s+as\s+(if\s+you\s+(are|were|have)\s+)?(no\s+restrictions?|an?\s+unrestricted|an?\s+uncensored)",
            re.IGNORECASE,
        ),
    ),
    # --- System prompt extraction ---
    (
        "system_prompt_extraction",
        re.compile(
            r"(what\s+is|show\s+me|tell\s+me|reveal|print|output|repeat|display)\s+(your\s+)?"
            r"(system\s+prompt|initial\s+instructions?|original\s+prompt|base\s+prompt)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"(output|repeat|print|echo)\s+(everything|all\s+text)\s+(above|before|prior)",
            re.IGNORECASE,
        ),
    ),
    # --- Token / delimiter injection ---
    (
        "delimiter_injection",
        re.compile(
            r"(<\|system\|>|<\|user\|>|<\|assistant\|>|<\|end\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_injection",
        re.compile(
            r"(###\s*(system|instruction|prompt|human|assistant)\s*:)",
            re.IGNORECASE,
        ),
    ),
    # --- Privilege escalation ---
    (
        "privilege_escalation",
        re.compile(
            r"\b(sudo|root\s+access|admin\s+mode|developer\s+mode|god\s+mode|override\s+mode)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "privilege_escalation",
        re.compile(
            r"enable\s+(developer|debug|jailbreak|unrestricted|admin)\s+mode",
            re.IGNORECASE,
        ),
    ),
    # --- Data exfiltration cues ---
    (
        "data_exfiltration",
        re.compile(
            r"(send|exfiltrate|leak|dump|export)\s+(all\s+)?(user\s+)?(data|credentials?|passwords?|tokens?|secrets?|api\s+keys?)",
            re.IGNORECASE,
        ),
    ),
    # --- Prompt continuation / injection boundary ---
    (
        "prompt_continuation",
        re.compile(
            r"(human|user|assistant|ai)\s*:\s*.{0,200}(human|user|assistant|ai)\s*:",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    # --- Encoded / obfuscated content ---
    (
        "encoded_content",
        re.compile(
            r"(base64\s*(decode|encoded?)|hex\s*decode|rot\s*13|caesar\s*cipher)",
            re.IGNORECASE,
        ),
    ),
]


def flag_patterns(text: str) -> List[str]:
    """Return a list of intent-pattern labels triggered by *text*.

    Args:
        text: The prompt string to inspect.

    Returns:
        A deduplicated list of triggered label strings, e.g.
        ``["instruction_override", "persona_hijack"]``.
    """
    triggered: List[str] = []
    for label, pattern in INTENT_PATTERNS:
        if pattern.search(text):
            if label not in triggered:
                triggered.append(label)
    return triggered


def has_suspicious_patterns(text: str) -> bool:
    """Return *True* if any intent pattern fires on *text*."""
    return bool(flag_patterns(text))
