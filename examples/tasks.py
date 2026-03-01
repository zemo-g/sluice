"""Example task functions built on sluice.

These show how to compose sluice primitives into domain-specific tasks.
Import the singleton client and call query() with appropriate model routing.
"""

from sluice import sluice


def summarize(text: str, max_bullets: int = 5) -> str:
    """Summarize text into bullet points using the fast model.

    Returns summary string or '' on failure.
    """
    system = (
        f"Summarize the following text into at most {max_bullets} bullet points. "
        "Focus on key facts and actionable items. Be concise."
    )
    return sluice.query("fast", text[:6000], system=system, max_tokens=1024)


def classify_sentiment(text: str) -> str:
    """Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL.

    Uses the fast model for sub-second classification.
    Returns one of: POSITIVE, NEGATIVE, NEUTRAL, UNKNOWN.
    """
    system = "Classify the sentiment of this text. Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL."
    result = sluice.query("fast", text, system=system, max_tokens=10, temperature=0.0)
    result = result.strip().upper()
    if result in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        return result
    return "UNKNOWN"


def code_review(filepath: str, code: str) -> str:
    """Review code for issues using the reasoning model.

    Returns structured findings as JSON string, or '' on failure.
    """
    system = (
        "You are a code reviewer. Find: dead imports, type errors, bare excepts, "
        "hardcoded values, missing error handling, deprecated patterns. "
        'Return JSON array: [{"line":N,"issue":"...","severity":"low|medium|high","fix":"..."}] '
        "Return [] if clean."
    )
    prompt = f"File: {filepath}\n\n```\n{code}\n```"
    return sluice.query("reasoning", prompt, system=system, max_tokens=4096)
