"""Local-first vision LLM service.

Ollama handles image understanding and transcription. Tesseract/PyMuPDF remain
the local text-extraction path in question_paper_reader.py. This module keeps
the same public function signatures used by exams.py and processing.py.
"""
import base64
import hashlib
import io
import json
import random
import re
import time
from typing import Any

import requests

from app.core.config import settings

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None


class LLMError(Exception):
    """A provider or model response could not be used."""


PAPER_PROMPT = """Read this printed examination question paper. Return ONLY a JSON object:
{"title":"string","negative_marking":0,"questions":[{"number":1,"text":"question text and options","type":"MCQ, Numeric, FillBlank, ShortAnswer, TrueFalse, Formula, Date, Matching or Descriptive","marks":1,"correct_answer":""}]}
Rules:
- Preserve the question numbers printed on the paper.
- Use the marks printed for each question; if marks are printed once for a section, apply them to each question in that section.
- negative_marking is a negative number (example -0.5); use 0 if none is stated.
- Automatically classify each question: MCQ (options), FillBlank (visible blank/underscores or fill-in instruction), TrueFalse, Matching, Date, Formula, Numeric (calculation/result), ShortAnswer (brief factual response), or Descriptive (explain/discuss/justify/long response).
- Do not classify every non-MCQ question as Numeric. Preserve the actual answer format implied by the question.
- Do NOT output instructions such as 'Answer all questions', 'Show your calculations' or 'Each question carries 2 marks' as questions; use them only to fill marks.
- A bare calculation such as '4,678 + 2,145 = ____' is Numeric, not FillBlank. FillBlank is only for a sentence with a missing word or value.
- Include all MCQ options in text.
- Leave correct_answer empty unless the paper itself contains the answer.
- Do not invent missing questions or marks. Use marks 0 if not visible.
"""

PAPER_TEXT_PROMPT = PAPER_PROMPT + """
The following text was extracted locally from the question paper. OCR may contain small errors.
Use only the supplied text; do not invent questions.

TEXT:
"""

KEY_PROMPT = """Read this examination answer key. Return ONLY JSON:
{"answers":{"1":"B","2":"42"}}
Map each visible question number to its correct answer. Preserve the complete expected answer for each question, including words, phrases, numbers, formulas, dates, true/false values and accepted alternatives. Do not truncate answers. Do not guess unclear entries; omit them.
"""

SHEET_PROMPT = """You are transcribing a student's handwritten examination answer sheet.
Question context (JSON) is provided only to help associate visible answers with question numbers:
{questions}

Return ONLY one JSON object in this exact shape:
{{"roll_no":"string or null","roll_confidence":0.0,"name":"string or null",
 "answers":[{{"number":1,"answer":"string, empty string, or null","confidence":0.0}}],
 "notes":"string"}}

Rules:
- First inspect the whole page, especially the header area, for the student's full name and roll number. The form may say "Name", "Student Name", "Roll No", "Roll Number", or similar. Read the value written next to each label, not the label itself.
- Preserve leading zeroes in roll numbers (for example, "05" must remain "05"). Do not mistake the class, date, question number, or page number for the roll number.
- If the name or roll number is printed or handwritten and clearly legible, return it even if some answers are unclear. If not legible, return null and low confidence.
- Transcribe only writing actually visible on the uploaded sheet. Never invent, complete, or correct a student's answer.
- Do not use question wording to guess an answer. Context is only for matching question numbers.
- Preserve mathematical signs, decimal points, fractions and written values. Do not add units.
- For MCQs, return only the clearly selected option letter (A, B, C or D).
- For FillBlank, ShortAnswer, TrueFalse, Formula, Date and Matching questions, transcribe the student response as written, preserving words, numbers, signs, units and symbols.
- For a question whose type is Descriptive, transcribe the student's whole written answer word for word as plain text. Keep their spelling and wording; do not summarise, correct or improve it.
- A visibly blank/unattempted answer must be answer="" and confidence=1.
- If writing exists but cannot be read reliably, use answer=null and low confidence.
- Never assign writing to a question number unless the match is clear.
- Return one entry per supplied question, including blank ones.
- Do not assess correctness or award marks. Python scoring code does that separately.
- Confidence measures legibility only, not whether the answer is correct. Use a number from 0.0 to 1.0.
- If roll number is unclear, use null and low roll_confidence. If it is clearly visible, give roll_confidence at least 0.8.
- The response must include name and roll_no fields even when their value is null.
"""

def _extract_json(text: str) -> dict:
    """Parse JSON even if the model wraps it in a Markdown code fence."""
    if not isinstance(text, str) or not text.strip():
        raise LLMError("Ollama returned an empty response.")
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        # Find the first complete JSON object without accepting prose as data.
        start = cleaned.find("{")
        if start < 0:
            raise LLMError("The vision model did not return a JSON object.")
        depth, quoted, escaped, end = 0, False, False, None
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if quoted:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    quoted = False
            else:
                if ch == '"':
                    quoted = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        if end is None:
            raise LLMError("The vision model returned incomplete JSON.")
        try:
            value = json.loads(cleaned[start:end])
        except json.JSONDecodeError as exc:
            raise LLMError(f"The vision model returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("The vision model response must be a JSON object.")
    return value


def _resize_image_bytes(data: bytes, mime: str, max_dim: int | None = None) -> tuple[bytes, str]:
    """Create a smaller in-memory image for the vision model; never modify the upload."""
    if Image is None:
        # Resizing is an optimisation, not a reason to reject an otherwise valid upload.
        return data, mime
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = opened.copy()
            image.thumbnail(
                (int(max_dim or getattr(settings, "OLLAMA_MAX_IMAGE_DIMENSION", 1800)),) * 2,
                Image.Resampling.LANCZOS,
            )
            output = io.BytesIO()
            if mime == "image/png":
                image.save(output, format="PNG", optimize=True)
                return output.getvalue(), "image/png"
            # JPEG is smaller and supported by Ollama. Flatten transparency safely.
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            image.save(
                output,
                format="JPEG",
                quality=int(getattr(settings, "OLLAMA_JPEG_QUALITY", 88)),
                optimize=True,
            )
            return output.getvalue(), "image/jpeg"
    except Exception as exc:
        raise LLMError(f"Could not prepare uploaded image for Ollama: {exc}") from exc


def _image_payloads(data: bytes, mime: str, max_dim: int | None = None) -> list[str]:
    """Return optimised base64 image(s); the original uploaded bytes are untouched."""
    if mime == "application/pdf":
        if fitz is None:
            raise LLMError("PyMuPDF is required to send PDF pages to Ollama. Install pymupdf.")
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            images = []
            try:
                if len(doc) == 0:
                    raise LLMError("The uploaded PDF has no pages.")
                if len(doc) > 12:
                    raise LLMError(
                        f"This PDF has {len(doc)} pages. The local vision model accepts up to 12 pages per upload; split the PDF and upload the parts separately."
                    )
                max_dim = int(getattr(settings, "OLLAMA_MAX_IMAGE_DIMENSION", 1800))
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    raw = pix.tobytes("png")
                    prepared, _ = _resize_image_bytes(raw, "image/png", max_dim)
                    images.append(base64.b64encode(prepared).decode("ascii"))
            finally:
                doc.close()
            return images
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Could not convert PDF pages for Ollama: {exc}") from exc

    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise LLMError(f"Unsupported image type for Ollama: {mime}")
    prepared, _prepared_mime = _resize_image_bytes(data, mime, max_dim)
    return [base64.b64encode(prepared).decode("ascii")]


def _ollama_json(prompt: str, data: bytes | None = None, mime: str | None = None, max_dim: int | None = None,
                 model: str | None = None) -> dict:
    base_url = str(settings.OLLAMA_URL).rstrip("/")
    model = str(model or settings.OLLAMA_MODEL).strip()
    if not model:
        raise LLMError("OLLAMA_MODEL is empty. Set it in backend/.env.")
    images = _image_payloads(data, mime, max_dim) if data is not None else []
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    body = {
        "model": model,
        "messages": [message],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
    }
    retries = max(0, int(getattr(settings, "OLLAMA_RETRIES", 2)))
    retry_delay = max(0.0, float(getattr(settings, "OLLAMA_RETRY_DELAY_SECONDS", 3)))
    timeout_seconds = max(30, int(settings.OLLAMA_TIMEOUT_SECONDS))
    response = None
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json=body,
                timeout=(10, timeout_seconds),
            )
            # Temporary server overloads are retried; other 4xx errors are final.
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            last_error = LLMError(
                f"Ollama temporarily returned HTTP {response.status_code}: {response.text[:400]}"
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        if attempt < retries:
            time.sleep(retry_delay * (attempt + 1))

    if response is None:
        raise LLMError(
            f"Ollama did not respond after {retries + 1} attempts. "
            f"Check that Ollama is running at {base_url}. Last error: {last_error}"
        ) from last_error
    if last_error is not None and response.status_code in (429, 500, 502, 503, 504):
        raise LLMError(
            f"Ollama remained unavailable after {retries + 1} attempts: {last_error}"
        )
    if response.status_code != 200:
        detail = response.text[:700]
        raise LLMError(f"Ollama returned HTTP {response.status_code}: {detail}")
    try:
        payload = response.json()
        content = payload.get("message", {}).get("content", "")
    except (ValueError, AttributeError) as exc:
        raise LLMError(f"Ollama returned an invalid API response: {exc}") from exc
    return _extract_json(content)


# Optional legacy provider retained so changing providers does not break an
# existing .env. New installations should use LLM_PROVIDER=ollama.
def _gemini_json(prompt: str, data: bytes | None = None, mime: str | None = None) -> dict:
    api_key = getattr(settings, "GEMINI_API_KEY", "")
    if not api_key:
        raise LLMError("Gemini is selected but GEMINI_API_KEY is not configured. Set LLM_PROVIDER=ollama.")
    model = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
    parts = [{"text": prompt}]
    if data is not None:
        parts.insert(0, {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode("ascii")}})
    body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            json=body, headers={"x-goog-api-key": api_key}, timeout=(15, 180)
        )
    except requests.RequestException as exc:
        raise LLMError(f"Gemini request failed: {exc}") from exc
    if response.status_code != 200:
        raise LLMError(f"Gemini request failed ({response.status_code}): {response.text[:500]}")
    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"Gemini returned an unreadable response: {exc}") from exc
    return _extract_json(text)


def _call(prompt: str, data: bytes | None = None, mime: str | None = None, max_dim: int | None = None) -> dict:
    provider = str(settings.LLM_PROVIDER).lower()
    if provider == "ollama":
        return _ollama_json(prompt, data, mime, max_dim)
    if provider == "gemini":
        return _gemini_json(prompt, data, mime)
    raise LLMError(f"Unsupported LLM_PROVIDER '{provider}'. Use 'ollama', 'gemini', or 'mock'.")


# ------------------------------------------------------------------ mock provider (deterministic fake data for development)
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
    return {"roll_no": None if bad_roll else str(2400 + rnd.randint(1, 99)),
            "roll_confidence": 0.3 if bad_roll else 0.95, "name": rnd.choice(names),
            "answers": answers, "notes": "mock data"}


def parse_paper(data: bytes, mime: str) -> dict:
    if settings.LLM_PROVIDER == "mock":
        return _mock_paper()
    return _call(PAPER_PROMPT, data, mime)


def parse_paper_text(text: str) -> dict:
    if settings.LLM_PROVIDER == "mock":
        return _mock_paper()
    return _call(PAPER_TEXT_PROMPT + text[:60000])


def parse_key_image(data: bytes, mime: str, numbers: list[int] | None = None) -> dict:
    if settings.LLM_PROVIDER == "mock":
        return {"answers": {str(q["number"]): q["correct_answer"] for q in mock_questions()}}
    result = _call(KEY_PROMPT, data, mime)
    answers = result.get("answers")
    if not isinstance(answers, dict):
        raise LLMError("The model did not return an 'answers' object for the answer key.")
    if numbers:
        allowed = {str(n) for n in numbers}
        answers = {str(k): v for k, v in answers.items() if str(k) in allowed}
    return {"answers": {str(k): str(v).strip() for k, v in answers.items() if v is not None and str(v).strip()}}


def parse_sheet(data: bytes, mime: str, questions: list[dict], max_dim: int | None = None) -> dict:
    if settings.LLM_PROVIDER == "mock":
        return _mock_sheet(data, questions)
    context = [{"number": int(q.get("number")), "type": str(q.get("type") or "MCQ"),
                "text": str(q.get("text") or "")[:500]} for q in questions]
    result = _call(SHEET_PROMPT.format(questions=json.dumps(context, ensure_ascii=False)), data, mime, max_dim)
    if not isinstance(result.get("answers"), list):
        raise LLMError("The model did not return an answers list for the student sheet.")
    return result
