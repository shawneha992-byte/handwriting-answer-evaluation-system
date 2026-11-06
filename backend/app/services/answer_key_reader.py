"""Answer-key extraction: local Tesseract OCR first, then local Ollama vision.

Supports:
- numbered keys: 1. 3601 / Q1: 3601
- solved worksheets: Answer = 3,601
- photographed/scanned JPG, PNG and PDF pages

This module only extracts answer text. It does not score student submissions.
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import fitz
import pytesseract
import requests
from PIL import Image, ImageOps, ImageEnhance

from app.core.config import settings


class AnswerKeyReadError(Exception):
    """Raised when neither local OCR nor the configured local vision model can read a key."""


# Explicitly capture final answer lines before considering numbered key lines.
ANSWER_LINE_RE = re.compile(
    r"""^\s*(?:final\s+)?answer\s*(?:is|:|=|-)\s*(.*?)\s*$""",
    re.IGNORECASE,
)
NUMBERED_LINE_RE = re.compile(
    r"""^\s*(?:Q(?:uestion)?\s*)?(\d{1,3})\s*[.)\:]\s*(.*?)\s*$""",
    re.IGNORECASE,
)
BARE_Q_RE = re.compile(
    r"""^\s*Q(?:uestion)?\s*(\d{1,3})\s*[-:]\s*(.*?)\s*$""",
    re.IGNORECASE,
)


def _clean_answer(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^(?:answer\s*[:=]\s*)", "", value, flags=re.I)
    value = value.strip(" \t;|")
    # OCR often inserts spaces around thousands separators.
    value = re.sub(r"(?<=\d)\s*,\s*(?=\d)", "", value)
    value = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_answers_from_text(text: str) -> dict[str, str]:
    """Extract explicit final-answer lines and simple numbered answer-key rows."""
    answers: dict[str, str] = {}
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]

    # First pass: explicit "Answer = ..." lines. Associate with the most recent Q number.
    current_q: str | None = None
    for line in lines:
        qmatch = re.search(r"\bQ(?:uestion)?\s*(\d{1,3})\b", line, re.I)
        if qmatch:
            current_q = str(int(qmatch.group(1)))

        match = ANSWER_LINE_RE.match(line)
        if match and current_q:
            value = _clean_answer(match.group(1))
            if value:
                answers[current_q] = value

    # Second pass: numbered key formats, but never treat a worked equation as an answer.
    for line in lines:
        match = NUMBERED_LINE_RE.match(line) or BARE_Q_RE.match(line)
        if not match:
            continue
        number, raw_value = match.groups()
        value = _clean_answer(raw_value)
        if not value or re.search(r"[=+*/]", value):
            continue
        # Ignore prose, question statements, and long workings.
        if len(value) > 80:
            continue
        # Explicit answer lines win over a nearby numbered question statement.
        answers.setdefault(str(int(number)), value)

    return answers


def _configure_tesseract() -> None:
    command = str(getattr(settings, "TESSERACT_CMD", "") or "").strip()
    if command:
        pytesseract.pytesseract.tesseract_cmd = command


def _image_pages(data: bytes, suffix: str) -> list[Image.Image]:
    if suffix == ".pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        pages: list[Image.Image] = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        return pages
    return [Image.open(io.BytesIO(data)).convert("RGB")]


def _prepare_for_ocr(image: Image.Image) -> Image.Image:
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    # Enlarge small scans for better character recognition.
    if image.width < 1800:
        scale = min(2.0, 1800 / max(1, image.width))
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    return image


def read_with_tesseract(data: bytes, suffix: str) -> tuple[str, str | None]:
    """Return (OCR text, error). Empty OCR is not treated as an exception."""
    try:
        _configure_tesseract()
        pages = _image_pages(data, suffix)
        texts = []
        for page in pages[:10]:
            prepared = _prepare_for_ocr(page)
            texts.append(pytesseract.image_to_string(prepared, config="--psm 6"))
        return "\n".join(texts), None
    except pytesseract.pytesseract.TesseractNotFoundError:
        return "", "Tesseract executable was not found. Install Tesseract or set TESSERACT_CMD in backend/.env."
    except Exception as exc:
        return "", f"Local OCR could not read this file: {exc}"


OLLAMA_KEY_PROMPT = """Read this teacher's answer key or solved worksheet.
Return ONLY a JSON object in this exact format:
{"answers":{"1":"3601","2":"6823"}}
Rules:
- Extract the FINAL answer for each question, not intermediate calculations.
- Understand both numbered key rows such as '1. 3601' and solved worksheets with lines such as 'Answer = 3,601'.
- If a question has working and a clearly labelled final 'Answer', use that final answer.
- Preserve negative signs and decimal points. Remove thousands separators from numeric answers.
- Do not invent or calculate missing answers. Omit anything you cannot read confidently.
- Keys must be question numbers as strings. Values must be strings.
"""


def read_with_ollama(data: bytes, suffix: str) -> dict[str, str]:
    base_url = str(getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    model = str(getattr(settings, "OLLAMA_MODEL", "qwen3-vl:4b-instruct"))
    try:
        pages = _image_pages(data, suffix)
        # Keep requests bounded: first 6 pages, resized to reduce local RAM and payload.
        images = []
        for page in pages[:6]:
            page.thumbnail((1600, 2200))
            buf = io.BytesIO()
            page.save(buf, format="JPEG", quality=85)
            images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        if not images:
            raise AnswerKeyReadError("The uploaded file contains no readable pages.")

        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": OLLAMA_KEY_PROMPT,
                "images": images,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=(10, 240),
        )
        if response.status_code != 200:
            raise AnswerKeyReadError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
            )
        payload = response.json()
        raw = payload.get("response", "")
        parsed = json.loads(raw)
        raw_answers = parsed.get("answers", {})
        if not isinstance(raw_answers, dict):
            raise AnswerKeyReadError("Ollama did not return an answers object.")
        return {
            str(k): _clean_answer(str(v))
            for k, v in raw_answers.items()
            if str(k).isdigit() and str(v).strip()
        }
    except AnswerKeyReadError:
        raise
    except requests.RequestException as exc:
        raise AnswerKeyReadError(
            f"Could not connect to Ollama at {base_url}. Make sure Ollama is running and the model '{model}' is downloaded. Details: {exc}"
        ) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise AnswerKeyReadError(f"Ollama returned invalid answer-key JSON: {exc}") from exc


def extract_answer_key(data: bytes, suffix: str) -> tuple[dict[str, str], str, str | None]:
    """Try Tesseract first; use Ollama vision if OCR misses the key."""
    ocr_text, ocr_error = read_with_tesseract(data, suffix)
    answers = extract_answers_from_text(ocr_text)
    if answers:
        return answers, "tesseract", ocr_error

    try:
        answers = read_with_ollama(data, suffix)
        if answers:
            return answers, "ollama-vision", ocr_error
        raise AnswerKeyReadError("Ollama could not identify any final answers in this file.")
    except AnswerKeyReadError as exc:
        if ocr_error:
            raise AnswerKeyReadError(f"{ocr_error} {exc}") from exc
        raise
