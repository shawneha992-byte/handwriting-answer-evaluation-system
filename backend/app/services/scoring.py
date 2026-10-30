"""Marks are calculated here, in plain code (not by the LLM), so results are repeatable."""
import re

from app.core.config import settings

NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def norm_mcq(value: str) -> str:
    s = re.sub(r"\b(OPTION|ANSWER|ANS)\b", "", str(value).upper())
    s = re.sub(r"[\s()\[\].:\-]", "", s)
    return s


def to_number(value: str) -> float | None:
    m = NUM.search(str(value).replace(",", ""))
    if not m:
        return None
    t = m.group(0)
    if "/" in t:
        a, b = t.split("/")
        return float(a) / float(b) if float(b) else None
    return float(t)


def is_correct(qtype: str, given: str, key: str) -> bool:
    if qtype.lower() == "numeric":
        a, b = to_number(given), to_number(key)
        return a is not None and b is not None and abs(a - b) <= settings.NUMERIC_TOLERANCE
    return norm_mcq(given) == norm_mcq(key)


def score_answer(question, given, confidence, negative_marking: float) -> tuple[str, float]:
    """Return (result, marks). Result: correct | wrong | unanswered | unreadable."""
    if given is None or (confidence is not None and confidence < settings.MIN_CONFIDENCE):
        return "unreadable", 0.0
    if str(given).strip() == "":
        return "unanswered", 0.0
    if is_correct(question.type, given, question.correct_answer):
        return "correct", float(question.marks)
    return "wrong", float(negative_marking)  # negative_marking is stored as a negative number (or 0)
