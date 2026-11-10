"""Deterministic, type-aware scoring. AI transcribes; this module awards marks."""
import re
import unicodedata
from datetime import datetime
from app.core.config import settings

NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")

def normalize_text(value: str) -> str:
    s = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    s = re.sub(r"^(?:answer|ans|option)\s*[:=.)-]?\s*", "", s)
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()

def norm_mcq(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper().replace("OPTION", "").replace("ANSWER", "").replace("ANS", ""))

def to_number(value: str) -> float | None:
    raw = str(value or "").replace(",", "")
    m = NUM.search(raw)
    if not m: return None
    t = m.group(0)
    try:
        if "/" in t:
            a,b=t.split("/"); return float(a)/float(b) if float(b) else None
        return float(t)
    except (ValueError, ZeroDivisionError): return None

def _date(value: str):
    s=str(value or "").strip()
    for fmt in ("%d/%m/%Y","%d-%m-%Y","%Y-%m-%d","%d.%m.%Y","%B %d, %Y","%d %B %Y","%b %d, %Y"):
        try: return datetime.strptime(s,fmt).date()
        except ValueError: pass
    return None

def _alternatives(key: str) -> list[str]:
    # Teacher may enter accepted alternatives separated by ||, |, or semicolon.
    return [x.strip() for x in re.split(r"\s*(?:\|\||\||;)\s*", str(key or "")) if x.strip()] or [str(key or "").strip()]

def is_correct(qtype: str, given: str, key: str) -> bool:
    typ=str(qtype or "").casefold().replace("_", "")
    candidates=_alternatives(key)
    if typ in {"numeric","number"}:
        a=to_number(given)
        return a is not None and any((b:=to_number(k)) is not None and abs(a-b)<=settings.NUMERIC_TOLERANCE for k in candidates)
    if typ in {"truefalse","boolean"}:
        aliases={"true":"true","t":"true","yes":"true","1":"true","false":"false","f":"false","no":"false","0":"false"}
        a=aliases.get(normalize_text(given),normalize_text(given))
        return any(a==aliases.get(normalize_text(k),normalize_text(k)) for k in candidates)
    if typ in {"date"}:
        a=_date(given)
        return a is not None and any(_date(k)==a for k in candidates if _date(k) is not None)
    if typ in {"formula","equation"}:
        compact=lambda x: re.sub(r"\s+", "", unicodedata.normalize("NFKC",str(x or "")).casefold()).replace("×","*").replace("÷","/").replace("−","-")
        return any(compact(given)==compact(k) for k in candidates)
    if typ in {"mcq"}:
        a=norm_mcq(given)
        return bool(a) and any(a==norm_mcq(k) for k in candidates)
    # FillBlank, ShortAnswer, Matching and textual key types: punctuation/case-insensitive exact alternatives.
    a=normalize_text(given)
    return bool(a) and any(a==normalize_text(k) for k in candidates)

def score_answer(question, given, confidence, negative_marking: float) -> tuple[str, float]:
    if given is None or (confidence is not None and confidence < settings.MIN_CONFIDENCE): return "unreadable", 0.0
    if str(given).strip()=="": return "unanswered", 0.0
    if is_correct(question.type, given, question.correct_answer): return "correct", float(question.marks)
    return "wrong", float(negative_marking)
