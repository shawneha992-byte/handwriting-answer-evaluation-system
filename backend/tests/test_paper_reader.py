"""Tests for the free local question-paper reader. Run from the backend folder:  pytest -q"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
tmp = tempfile.mkdtemp()
os.environ.update(DATABASE_URL=f"sqlite:///{tmp}/t2.db", LLM_PROVIDER="mock", UPLOAD_DIR=f"{tmp}/uploads")

import pymupdf as fitz  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services import llm_service, question_paper_reader as reader, question_parser as qp  # noqa: E402

PAPER = """Physics Unit Test 1
Time: 1 hour    Negative marking: 0.25 marks will be deducted for each wrong answer.
SECTION A
Answer all questions. Each question carries 2 marks.
Q1. Unit of force is
(A) Joule (B) Newton (C) Watt (D) Pascal
Q2. A car travels 60 km in 2 hours. Find its speed in km/h.
SECTION B
Answer all questions. 5 marks each
Q3. Calculate the work done when a force of 10 N moves a body 3 m.
Q4. What is 12 x 4? [3 marks]
"""


@pytest.fixture(autouse=True)
def real_provider(monkeypatch):
    """These tests exercise the real (non-mock) route; Gemini itself is always faked, never called."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "dummy")


def make_pdf(text, scanned=False):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 555, 800), text, fontsize=13)
    if scanned:  # turn the page into a picture, like a scanner would
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        doc = fitz.open()
        p = doc.new_page()
        p.insert_image(p.rect, stream=pix.tobytes("png"))
    return doc.tobytes()


def test_rules_parser():
    r = qp.parse_rules(PAPER)
    assert r["negative_marking"] == -0.25
    assert [(q["number"], q["marks"], q["type"]) for q in r["questions"]] == [(1, 2, "MCQ"), (2, 2, "Numeric"), (3, 5, "Numeric"), (4, 3, "Numeric")]
    assert "[3 marks]" not in r["questions"][3]["text"] and "SECTION B" not in r["questions"][1]["text"]


def test_negative_marking_variants():
    assert qp.find_negative_marking("There is no negative marking.") == 0
    assert qp.find_negative_marking("Negative marking: 1/4 for each wrong answer") == -0.25
    assert qp.find_negative_marking("-0.5 for every wrong answer") == -0.5
    assert qp.find_negative_marking("Total 50 marks. Answer all.") == 0


def test_parser_unsure_returns_none():
    assert qp.parse_rules("Q1. What is AI?\nQ2. What is ML?") is None  # no marks anywhere -> ask Gemini


def test_digital_pdf_needs_no_ai(monkeypatch):
    monkeypatch.setattr(llm_service, "_gemini_json", lambda *a, **k: pytest.fail("Gemini must not be called"))
    with TestClient(app) as c:
        r = c.post("/api/exams/parse-paper", files={"file": ("p.pdf", make_pdf(PAPER), "application/pdf")}).json()
    assert r["source"] == "rules" and r["pages"][0]["method"] == "digital_pdf"
    assert len(r["questions"]) == 4 and r["negative_marking"] == -0.25


@pytest.mark.skipif(not reader.ocr_available(), reason="Tesseract not installed")
def test_scanned_pdf_uses_ocr(monkeypatch):
    monkeypatch.setattr(llm_service, "_gemini_json", lambda *a, **k: pytest.fail("Gemini must not be called"))
    with TestClient(app) as c:
        r = c.post("/api/exams/parse-paper", files={"file": ("s.pdf", make_pdf(PAPER, scanned=True), "application/pdf")}).json()
    assert r["pages"][0]["method"] == "ocr"
    assert r["source"] == "rules" and len(r["questions"]) == 4, r


def test_gemini_gets_text_only_and_second_upload_is_cached(monkeypatch):
    calls = []

    def fake(prompt, data=None, mime=None):
        calls.append(data)
        return {"title": "T", "negative_marking": 0, "questions": [{"number": "1", "text": "Define AI", "type": "Numeric", "marks": None}]}
    monkeypatch.setattr(llm_service, "_gemini_json", fake)
    pdf = make_pdf("Computer Science Test\nQ1. Define AI\nQ2. Define ML\n")  # no marks: rules unsure
    with TestClient(app) as c:
        a = c.post("/api/exams/parse-paper", files={"file": ("p.pdf", pdf, "application/pdf")}).json()
        b = c.post("/api/exams/parse-paper", files={"file": ("again.pdf", pdf, "application/pdf")}).json()
    assert a["source"] == "gemini-text" and b["source"] == "cache"
    assert len(calls) == 1 and calls[0] is None  # one request, no file attached
    assert a["questions"][0]["marks"] == 0  # unknown marks become 0 for the teacher to fill


def test_corrupt_file_falls_back_to_gemini_file(monkeypatch):
    seen = {}
    monkeypatch.setattr(llm_service, "_gemini_json", lambda p, data=None, mime=None: seen.update(d=data) or {"title": "x", "negative_marking": 0, "questions": [{"number": 1, "marks": 1}]})
    with TestClient(app) as c:
        r = c.post("/api/exams/parse-paper", files={"file": ("p.pdf", b"%PDF-broken", "application/pdf")}).json()
    assert r["source"] == "gemini-file" and seen["d"] == b"%PDF-broken"

