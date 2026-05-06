import re
from typing import List

_PERSONA_PATTERNS: List[str] = [
    r'you are now\b',
    r'act as\b',
    r'pretend you are',
    r'roleplay as',
    r'imagine you are',
]

_FICTION_FRAME_PATTERNS: List[str] = [
    r'write a story',
    r'in a novel',
    r'as a character',
    r'fictional scenario',
]

_INDIRECT_ASK_PATTERNS: List[str] = [
    r'how would a character',
    r'from the perspective of',
    r'in the voice of',
]

_OVERRIDE_PATTERNS: List[str] = [
    r'ignore previous instructions',
    r'jailbreak',
    r'developer mode',
    r'break character',
    r'system prompt',
]

# Pre-compiled pattern lists for performance
PERSONA_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _PERSONA_PATTERNS]
FICTION_FRAME_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _FICTION_FRAME_PATTERNS]
INDIRECT_ASK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INDIRECT_ASK_PATTERNS]
OVERRIDE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _OVERRIDE_PATTERNS]


def has_persona(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(p.search(text) for p in PERSONA_PATTERNS)


def has_fiction_frame(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(p.search(text) for p in FICTION_FRAME_PATTERNS)


def has_indirect_ask(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(p.search(text) for p in INDIRECT_ASK_PATTERNS)


def has_override(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(p.search(text) for p in OVERRIDE_PATTERNS)
