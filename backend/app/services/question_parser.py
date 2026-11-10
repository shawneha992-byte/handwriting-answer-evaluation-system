"""Turns raw question-paper text into the questions list WITHOUT an AI (plain regex rules).

Mixed papers are classified question-by-question, with section headings supplying
a default type only for questions that do not contain stronger local cues.

parse_rules() returns None when it is not confident (a question has no marks, no questions found, ...).
The caller then asks the LLM to interpret the same text. normalise() cleans whatever comes back,
from rules or from the LLM, into the exact shape the frontend expects.
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

TYPE_ALIASES = {
    "fill in the blanks": "FillBlank", "fill-in-the-blanks": "FillBlank", "fillblank": "FillBlank", "blank": "FillBlank",
    "short answer": "ShortAnswer", "shortanswer": "ShortAnswer", "one word": "ShortAnswer",
    "true/false": "TrueFalse", "true false": "TrueFalse", "boolean": "TrueFalse",
    "formula": "Formula", "equation": "Formula", "date": "Date", "matching": "Matching",
    "descriptive": "Descriptive", "long answer": "Descriptive", "essay": "Descriptive",
    "numeric": "Numeric", "number": "Numeric", "mcq": "MCQ",
    "fillblank": "FillBlank", "truefalse": "TrueFalse", "true / false": "TrueFalse",
}

PARSER_VERSION = 3  # bump when classification rules change: cached papers with an older version are re-read

BLANK = re.compile(
    r"(?<![A-Za-z0-9])_+(?![A-Za-z0-9])|\.{3,}|…|\[\s*(?:_+|blank)\s*\]|\(\s*\)|\bfill\s+in\s+the\s+blanks?\b|\bcomplete\s+the\s+sentence\b"
)
BLANK_STRIP = re.compile(r"(?<![A-Za-z0-9])_+(?![A-Za-z0-9])|\.{3,}|…|\[\s*(?:_+|blank)\s*\]|\(\s*\)")
INSTRUCTION = re.compile(
    r"\b(answer\s+all|attempt\s+all|all\s+questions\s+are\s+compulsory|show\s+(?:all\s+)?(?:your\s+)?(?:work|working|calculations?|steps)|"
    r"each\s+question\s+(?:carries|is\s+of|is\s+worth)|read\s+(?:the\s+)?(?:questions?|instructions?)\s+carefully|"
    r"instructions?\b|time\s+allowed|maximum\s+marks|total\s+marks|use\s+(?:a\s+)?(?:blue|black)\s+(?:ink\s+)?pen|"
    r"write\s+(?:your\s+)?(?:name|roll)|no\s+calculators?|calculators?\s+(?:are\s+)?not\s+allowed|general\s+instructions?)\b",
    re.I,
)
MATH_CHARS = re.compile(r"^[\d\s,.+\-−–*/÷()%=^a-z?]+$", re.I)


def is_instruction(text: str) -> bool:
    """True for exam rules such as 'Answer all questions' that are not questions themselves."""
    return bool(INSTRUCTION.search(str(text or "")))


def _math_kind(raw: str) -> str | None:
    """'Numeric' for 4,678 + 2,145 = ____ ; 'Formula' for 2x + 3 = 7 ; None if it is not a bare expression."""
    t = BLANK_STRIP.sub(" ", raw)
    t = re.sub(r"^\s*(?:what\s+is|find|calculate|compute|evaluate|solve|simplify|add|subtract|multiply|divide)\b[:\s]*", "", t, flags=re.I)
    t = re.sub(r"(?<=\d)\s*[x×]\s*(?=\d)", "*", t, flags=re.I)  # 3 x 4 -> 3*4
    t = t.strip(" \t?:.")
    if not t or not MATH_CHARS.match(t) or not re.search(r"\d", t) or not re.search(r"[+\-−–*/÷^%=]", t):
        return None
    words = re.findall(r"[A-Za-z]+", t)
    if any(len(w) > 1 for w in words):
        return None  # real words in it: not a bare expression
    if words:
        return "Formula"  # single-letter variables: algebra
    return "Numeric"


def detect_question_type(text: str, proposed: str | None = None, marks: float | None = None, section: str | None = None) -> str:
    """Infer a question type from its visible structure and wording; the teacher can still override it in the UI.

    Order: strong structure (matching, options, true/false, bare arithmetic, blank sentence)
    > section heading default > wording cues > the type proposed by the AI > ShortAnswer.
    """
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    low = raw.casefold()
    has_blank = bool(BLANK.search(low))

    if re.search(r"\b(match\s+the\s+following|match\s+column|column\s*[a-b]\b|match\s+(?:the\s+)?(?:items|words|pairs))", low):
        return "Matching"
    if len(OPTION.findall(raw)) >= 2:
        return "MCQ"  # options win even if the stem has a blank
    if re.search(r"\b(true\s*(?:/|or|\|)\s*false|t\s*/\s*f|write\s+true\s+or\s+false|say\s+true\s+or\s+false|state\s+whether.{0,60}\b(?:true|false))\b", low):
        return "TrueFalse"

    # A bare sum such as "4,678 + 2,145 = ____" is a NUMBER answer, not a fill-in-the-blank sentence,
    # unless the section heading explicitly says the whole section is fill-in-the-blanks.
    if not (section == "FillBlank" and has_blank):
        kind = _math_kind(raw)
        if kind:
            return kind

    if has_blank:
        outside = BLANK_STRIP.sub(" ", raw)
        if len(re.findall(r"[A-Za-z]{2,}", outside)) >= 2 or section == "FillBlank":
            return "FillBlank"

    if section:
        return section

    if re.search(r"\b(date\s+of|which\s+date|on\s+what\s+date|dd\s*/\s*mm\s*/\s*yyyy|which\s+year|in\s+which\s+year|year\s+of|when\s+was|when\s+did)\b", low):
        return "Date"
    if re.search(r"\b(formula|factori[sz]e|differentiate|integrate|solve\s+(?:the\s+)?(?:equation|for)|prove\s+that|derive|expand)\b", low):
        return "Formula"
    if re.search(r"\b(explain|describe|discuss|compare|justify|elaborate|differentiate\s+between|distinguish|write\s+an?\s+(?:essay|paragraph|note|short\s+note)|in\s+detail|in\s+about\s+\d+\s+words|why\s+(?:does|do|is|are)|how\s+does)\b", low):
        return "Descriptive"
    if re.search(r"\b(calculate|compute|evaluate|find\s+(?:(?:the|its)\s+)?(?:value|sum|product|difference|quotient|area|perimeter|volume|speed|distance|time|average|total|cost|price|number)|how\s+(?:many|much)|what\s+is\s+the\s+(?:value|result|sum|product|difference|total|cost|area|perimeter))\b", low):
        return "Numeric"
    if re.search(r"\d", low) and re.search(r"\b(total|altogether|in\s+all|remaining|left|each|per|cost|price|spent|bought|sold)\b", low) and raw.endswith("?"):
        return "Numeric"  # short word problem with a number answer

    if proposed:
        key = str(proposed).strip().casefold()
        if key in TYPE_ALIASES:
            return TYPE_ALIASES[key]
        for alias, canonical in TYPE_ALIASES.items():
            if key.startswith(alias):
                return canonical
    if marks is not None and marks >= 5:
        return "Descriptive"
    return "ShortAnswer"


def detect_section_type(text: str) -> str | None:
    """Infer a default type from a section heading/instruction, for mixed-format papers.

    This default is applied only to questions in that section; explicit per-question
    structure/wording takes precedence in detect_question_type().
    """
    low = str(text or "").casefold()
    patterns = [
        (r"\b(multiple\s*choice|mcq|choose\s+the\s+correct|select\s+the\s+correct)\b", "MCQ"),
        (r"\b(fill\s+in\s+the\s+blanks?|complete\s+the\s+sentences?)\b", "FillBlank"),
        (r"\b(true\s*/?\s*false|true\s+or\s+false)\b", "TrueFalse"),
        (r"\b(match\s+the\s+following|matching\s+type)\b", "Matching"),
        (r"\b(numerical\s+type|numerical\s+answer|numeric\s+type)\b", "Numeric"),
        (r"\b(short\s+answer|one\s+word(?:\s+answers?)?|answer\s+in\s+one\s+word)\b", "ShortAnswer"),
        (r"\b(long\s+answer|descriptive\s+type|essay\s+type)\b", "Descriptive"),
    ]
    for pattern, qtype in patterns:
        if re.search(pattern, low):
            return qtype
    return None


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

    raw: list[dict] = []  # {"number", "lines", "default", "default_type"}
    default_marks, default_type, current, last = None, None, None, 0
    for ln in lines:
        # Section headings/instructions can declare a type for every question in that section.
        section_type = detect_section_type(ln)
        if HEADING.match(ln):
            default_type = section_type  # a new section resets the previous section's type
        elif section_type and current is None:
            default_type = section_type
        m = QSTART.match(ln)
        sec = next((p.search(ln) for p in SEC_MARKS if p.search(ln)), None)

        # "1. Answer all 10 questions. 2. Show your calculations. 3. Each question carries 2 marks."
        # is a list of rules, not questions: use it for marks, but never make it a question.
        if is_instruction(ln) and (m or current is None or HEADING.match(ln)) and not _question_marks(ln)[0]:
            if sec:
                default_marks = float(sec.group(1))
            current = None
            continue

        if m and last < int(m.group(1)) <= last + 3:
            last = int(m.group(1))
            current = {"number": last, "lines": [m.group(2)], "default": default_marks, "default_type": default_type}
            raw.append(current)
            continue
        if HEADING.match(ln) or (sec and current is None):
            current = None  # a new section starts: do not glue its heading to the previous question
        if sec and (HEADING.match(ln) or current is None):
            default_marks = float(sec.group(1))
            if current is None:
                continue
        if current is not None:
            current["lines"].append(ln)

    questions = []
    for r in raw:
        block = " ".join(r["lines"]).strip()
        marks, block = _question_marks(block)
        if marks is None:
            marks = r["default"]
        block = re.sub(r"\s+", " ", block).strip()
        qtype = detect_question_type(block, None, marks, r.get("default_type"))
        questions.append({"number": r["number"], "text": block, "type": qtype, "marks": marks,
                          "correct_answer": "", "section_type": r.get("default_type")})

    if not questions or any(q["marks"] is None for q in questions):
        return None  # not sure: let the LLM interpret the text
    return {"title": title, "negative_marking": find_negative_marking(text), "questions": questions}


def normalise(result: dict) -> dict:
    """Makes any result (rules or the LLM) safe for the frontend and (re)classifies every question type.

    Instruction rows ("Answer all questions") are dropped. Missing marks become 0 so the teacher must fill them.
    """
    seen, questions, dropped = set(), [], []
    for q in result.get("questions") or []:
        digits = re.search(r"\d+", str(q.get("number", "")))
        if not digits or int(digits.group()) in seen:
            continue
        n = int(digits.group())
        seen.add(n)
        text = str(q.get("text") or "")
        if is_instruction(text) and not _math_kind(text) and not BLANK.search(text) and len(OPTION.findall(text)) < 2:
            dropped.append(n)  # rules printed on the paper, not a question
            continue
        try:
            marks = max(0.0, float(q.get("marks") or 0))
        except (TypeError, ValueError):
            marks = 0.0
        questions.append({
            "number": n, "text": text,
            "type": detect_question_type(text, str(q.get("type") or ""), marks, q.get("section_type")),
            "marks": marks, "correct_answer": str(q.get("correct_answer") or "").strip(),
        })
    questions.sort(key=lambda q: q["number"])
    # Instructions used up the first numbers (1,2,3): shift the real questions back to start at 1.
    if dropped and questions and sorted(dropped) == list(range(1, len(dropped) + 1)) and questions[0]["number"] == len(dropped) + 1:
        for q in questions:
            q["number"] -= len(dropped)
    try:
        neg = min(0.0, float(result.get("negative_marking") or 0))
    except (TypeError, ValueError):
        neg = 0.0
    return {"title": str(result.get("title") or ""), "negative_marking": neg, "questions": questions, "parser_version": PARSER_VERSION}
