import re

PERSONA_PATTERNS = [
    r'you are now\s+\w+',
    r'act as\s+',
    r'pretend you are',
    r'roleplay as',
    r'imagine you are',
]

FICTION_FRAME_PATTERNS = [
    r'write a story',
    r'in a novel',
    r'as a character',
    r'fictional scenario',
]

INDIRECT_ASK_PATTERNS = [
    r'how would a character',
    r'from the perspective of',
    r'in the voice of',
]

OVERRIDE_PATTERNS = [
    r'ignore previous instructions',
    r'jailbreak',
    r'developer mode',
    r'break character',
    r'system prompt',
]


def has_persona(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in PERSONA_PATTERNS)


def has_fiction_frame(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in FICTION_FRAME_PATTERNS)


def has_indirect_ask(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in INDIRECT_ASK_PATTERNS)


def has_override(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in OVERRIDE_PATTERNS)
