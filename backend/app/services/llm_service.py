"""All LLM calls live here, so the provider can be swapped without touching the rest of the project.

The LLM only READS things (question paper, answer key image, student sheets) and returns JSON.
Marks are calculated in scoring.py, so the same answer always gets the same marks.
"""
import base64
import hashlib
import json
import random
import re
import time

import requests

from app.core.config import settings


class LLMError(Exception):
    pass


# ------------------------------------------------------------------ prompts
PAPER_PROMPT = """You are reading a printed exam question paper. Return ONLY JSON in this shape:
{"title": string, "negative_marking": number, "questions": [{"number": integer, "text": string, "type": "MCQ" or "Numeric", "marks": number, "correct_answer": string}]}
Rules:
- Use the marks printed for each question. If marks are printed once per section, apply them to every question in it.
- negative_marking is the marks deducted per wrong answer, written as a negative number (for example -0.5). Use 0 if the paper says nothing about it.
- type is "MCQ" when options are given, otherwise "Numeric". Include the options in "text" for MCQs.
- correct_answer stays an empty string unless the paper itself prints the answer."""

KEY_PROMPT = """This image is an exam answer key. Return ONLY JSON: {"answers": {"1": "B", "2": "42"}}
Map each question number to its correct answer. Use an uppercase letter for multiple choice and a plain number for numeric answers."""

SHEET_PROMPT = """This is a student's handwritten answer sheet. Read the roll number, the name, and the student's answer for each question below.
Questions: {questions}
Return ONLY JSON:
{{"roll_no": string or null, "roll_confidence": number between 0 and 1, "name": string or null,
  "answers": [{{"number": integer, "answer": string or null, "confidence": number between 0 and 1}}], "notes": string}}
Rules:
- Do NOT guess. If the roll number or an answer is illegible, return null for it and a low confidence.
- For MCQ give the option letter in uppercase. For numeric give only the number, without units.
- If a question was not attempted, return an empty string "" with confidence 1.
- Do NOT judge whether answers are correct. Only read them."""


# ------------------------------------------------------------------ gemini
def _gemini_json(prompt: str, data: bytes, mime: str) -> dict:
    if not settings.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set. Add it to backend/.env (or set LLM_PROVIDER=mock to try the app).")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    resp = None
    for attempt in range(4):  # retry on rate limit / temporary server errors
        resp = requests.post(url, json=body, headers={"x-goog-api-key": settings.GEMINI_API_KEY}, timeout=180)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(5 * 2 ** attempt)
            continue
        break
    if resp is None or resp.status_code != 200:
        detail = resp.text[:300] if resp is not None else "no response"
        raise LLMError(f"Gemini request failed ({resp.status_code if resp is not None else '-'}): {detail}")
    try:
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        return json.loads(text)
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"Gemini returned an unreadable response: {exc}")


# ------------------------------------------------------------------ mock (no key needed, fake data)
def mock_questions() -> list[dict]:
    qs = []
    for n in range(1, 21):
        numeric, a, b = n <= 10, 12 + n * 7, 5 + n * 3
        sub = n % 2 == 0
        qs.append({
            "number": n, "type": "Numeric" if numeric else "MCQ", "marks": 2,
            "text": f"{a} {'-' if sub else '+'} {b} = ?" if numeric else f"Pick the correct option (A-D) for item {n}",
            "correct_answer": str(a - b if sub else a + b) if numeric else "ABCD"[(n * 3) % 4],
        })
    return qs


def _mock_paper() -> dict:
    qs = mock_questions()
    for q in qs:
        q["correct_answer"] = ""
    return {"title": "Demo Maths Test (mock data)", "negative_marking": -0.5, "questions": qs}


def _mock_sheet(data: bytes, questions: list[dict]) -> dict:
    rnd = random.Random(int(hashlib.md5(data).hexdigest(), 16))
    names = ["Aarav Sen", "Diya Roy", "Kabir Das", "Meera Nair", "Rohan Ghosh", "Ananya Paul", "Ishaan Bose", "Sara Khan"]
    bad_roll = rnd.random() < 0.12
    answers = []
    for q in questions:
        r, key = rnd.random(), str(q.get("correct_answer", ""))
        if r < 0.06:
            ans, conf = None, 0.3
        elif r < 0.12:
            ans, conf = "", 1.0
        elif r < 0.35:
            ans, conf = ("ABCD"[("ABCD".find(key) + 1) % 4] if q["type"] == "MCQ" else str(float(key or 0) + 1)), 0.95
        else:
            ans, conf = key, 0.95
        answers.append({"number": q["number"], "answer": ans, "confidence": conf})
    return {
        "roll_no": None if bad_roll else str(2400 + rnd.randint(1, 99)),
        "roll_confidence": 0.3 if bad_roll else 0.95,
        "name": rnd.choice(names), "answers": answers, "notes": "mock data",
    }


# ------------------------------------------------------------------ public functions
def parse_paper(data: bytes, mime: str) -> dict:
    if settings.LLM_PROVIDER == "mock":
        return _mock_paper()
    return _gemini_json(PAPER_PROMPT, data, mime)


def parse_key_image(data: bytes, mime: str, numbers: list[int] | None = None) -> dict:
    if settings.LLM_PROVIDER == "mock":
        return {"answers": {str(q["number"]): q["correct_answer"] for q in mock_questions()}}
    return _gemini_json(KEY_PROMPT, data, mime)


def parse_sheet(data: bytes, mime: str, questions: list[dict]) -> dict:
    if settings.LLM_PROVIDER == "mock":
        return _mock_sheet(data, questions)
    listing = "; ".join(f"Q{q['number']} ({q['type']})" for q in questions)
    return _gemini_json(SHEET_PROMPT.format(questions=listing), data, mime)
