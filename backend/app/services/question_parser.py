"""Turns raw question-paper text into the questions list WITHOUT an AI (plain regex rules).

parse_rules() returns None when it is not confident (a question has no marks, no questions found, ...).
The caller then asks Gemini to interpret the same text. normalise() cleans whatever comes back,
from rules or from Gemini, into the exact shape the frontend expects.
"""
import re

NUM = r"\d+(?:\.\d+)?"

# "Q1." "Q.1)" "Question 1:" "1." "1)" at the start of a line
QSTART = re.compile(r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[.):\-]\s*(.*)$", re.I)
HEADING = re.compile(r"^\s*(section|part)\b", re.I)

# marks printed on one question: [2]  (2 marks)  [2 Marks]  (2M)
Q_MARKS = [
    re.compile(rf"[\[(]\s*({NUM})\s*(?:marks?|mark|m)\s*[\])]", re.I),
    re.compile(rf"\[\s*({NUM})\s*\]\s*$"),
    re.compile(rf"\(\s*({NUM})\s*\)\s*$"),
]
# marks printed once for a section: "2 marks each", "each question carries 2 marks"
SEC_MARKS = [
    re.compile(rf"({NUM})\s*marks?\s*each", re.I),
    re.compile(rf"each\s+(?:question\s+)?(?:carries?|is\s+of|is\s+worth|of)\s*({NUM})\s*marks?", re.I),
    re.compile(rf"carr(?:y|ies)\s*({NUM})\s*marks?\s*each", re.I),
]
OPTION = re.compile(r"(?:^|\s)\(?[A-Da-d][).]\s")
FRACTION = re.compile(r"^(\d+)\s*/\s*(\d+)$")

NEG_NONE = re.compile(r"no\s+negative", re.I)
NEG_PATTERNS = [
    re.compile(r"(\d+\s*/\s*\d+|\d+(?:\.\d+)?)\s*marks?\s*(?:will\s+be\s+|shall\s+be\s+|is\s+|are\s+)?(?:deducted|penali[sz]ed)", re.I),
    re.compile(r"(?:negative\s+marking|penalty|deduct\w*)[^\n.]{0,60}?(\d+\s*/\s*\d+|\d+(?:\.\d+)?)", re.I),
    re.compile(r"[-\u2212\u2013]\s*(\d+\s*/\s*\d+|\d+(?:\.\d+)?)\s*(?:marks?\s*)?for\s+(?:each\s+|every\s+)?(?:wrong|incorrect)", re.I),
]


def _to_number(s: str) -> float:
    s = s.replace(" ", "")
    m = FRACTION.match(s)
    return int(m.group(1)) / int(m.group(2)) if m and int(m.group(2)) else float(s)


def find_negative_marking(text: str) -> float:
    """Marks deducted per wrong answer as a negative number, 0 if the paper says nothing."""
    if NEG_NONE.search(text):
        return 0.0
    for pat in NEG_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                v = _to_number(m.group(1))
            except ValueError:
                continue
            if 0 < v <= 5:  # ignore silly matches
                return -round(v, 4)
    return 0.0


def _question_marks(block: str):
    for pat in Q_MARKS:
        m = pat.search(block)
        if m:
            return float(m.group(1)), pat.sub("", block, count=1).strip()
    return None, block


def parse_rules(text: str):
    """Returns {"title", "negative_marking", "questions"} or None if not confident."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    title = next((ln for ln in lines if len(ln) <= 120 and not QSTART.match(ln) and re.search(r"[A-Za-z]{3}", ln)), "")

    raw: list[dict] = []  # {"number", "lines", "default"}
    default_marks, current, last = None, None, 0
    for ln in lines:
        m = QSTART.match(ln)
        if m and last < int(m.group(1)) <= last + 3:
            last = int(m.group(1))
            current = {"number": last, "lines": [m.group(2)], "default": default_marks}
            raw.append(current)
            continue
        sec = next((p.search(ln) for p in SEC_MARKS if p.search(ln)), None)
        if HEADING.match(ln) or (sec and current is None):
            current = None  # a new section starts: do not glue its heading to the previous question
        if sec and (HEADING.match(ln) or current is None):
            default_marks = float(sec.group(1))
            if current is None:
                continue
        if current is not None:
            current["lines"].append(ln)
        # lines before the first question (title, instructions) are only scanned for rules above

    questions = []
    for r in raw:
        block = " ".join(r["lines"]).strip()
        marks, block = _question_marks(block)
        if marks is None:
            marks = r["default"]
        block = re.sub(r"\s+", " ", block).strip()
        qtype = "MCQ" if len(OPTION.findall(block)) >= 2 else "Numeric"
        questions.append({"number": r["number"], "text": block, "type": qtype, "marks": marks, "correct_answer": ""})

    if not questions or any(q["marks"] is None for q in questions):
        return None  # not sure: let Gemini interpret the text
    return {"title": title, "negative_marking": find_negative_marking(text), "questions": questions}


def normalise(result: dict) -> dict:
    """Makes any result (rules or Gemini) safe for the frontend. Missing marks become 0 so the teacher must fill them."""
    seen, questions = set(), []
    for q in result.get("questions") or []:
        digits = re.search(r"\d+", str(q.get("number", "")))
        if not digits or int(digits.group()) in seen:
            continue
        n = int(digits.group())
        seen.add(n)
        try:
            marks = max(0.0, float(q.get("marks") or 0))
        except (TypeError, ValueError):
            marks = 0.0
        questions.append({
            "number": n, "text": str(q.get("text") or ""),
            "type": "Numeric" if str(q.get("type", "")).lower().startswith("num") else "MCQ",
            "marks": marks, "correct_answer": str(q.get("correct_answer") or "").strip(),
        })
    questions.sort(key=lambda q: q["number"])
    try:
        neg = min(0.0, float(result.get("negative_marking") or 0))
    except (TypeError, ValueError):
        neg = 0.0
    return {"title": str(result.get("title") or ""), "negative_marking": neg, "questions": questions}
