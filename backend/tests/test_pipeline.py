"""End-to-end test of the whole pipeline using the mock LLM and a throwaway SQLite database.
Run from the backend folder:  pytest -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
tmp = tempfile.mkdtemp()
os.environ.update(DATABASE_URL=f"sqlite:///{tmp}/test.db", LLM_PROVIDER="mock", UPLOAD_DIR=f"{tmp}/uploads")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import scoring  # noqa: E402
from app.services.llm_service import mock_questions  # noqa: E402


def test_scoring_rules():
    class Q:  # minimal question stand-in
        def __init__(self, t, k): self.type, self.correct_answer, self.marks = t, k, 2
    assert scoring.score_answer(Q("MCQ", "B"), "(b)", 0.9, -0.5) == ("correct", 2.0)
    assert scoring.score_answer(Q("MCQ", "B"), "Option C", 0.9, -0.5) == ("wrong", -0.5)
    assert scoring.score_answer(Q("Numeric", "42"), "42.0", 0.9, 0) == ("correct", 2.0)
    assert scoring.score_answer(Q("MCQ", "B"), "", 1, -0.5) == ("unanswered", 0.0)
    assert scoring.score_answer(Q("MCQ", "B"), None, 0.2, -0.5)[0] == "unreadable"
    assert scoring.score_answer(Q("MCQ", "B"), "B", 0.2, -0.5)[0] == "unreadable"


def test_full_pipeline():
    with TestClient(app) as c:
        paper = c.post("/api/exams/parse-paper", files={"file": ("paper.pdf", b"%PDF-demo", "application/pdf")}).json()
        assert len(paper["questions"]) == 20 and paper["negative_marking"] == -0.5

        csv_key = "Q,Answer\n" + "\n".join(f"{q['number']},{q['correct_answer']}" for q in mock_questions())
        key = c.post("/api/exams/parse-key", files={"file": ("key.csv", csv_key.encode(), "text/csv")}).json()["answers"]
        assert len(key) == 20

        qs = [{**q, "correct_answer": key[str(q["number"])]} for q in paper["questions"]]
        exam_id = c.post("/api/exams", json={"title": "T", "negative_marking": -0.5, "questions": qs}).json()["id"]
        assert c.post("/api/exams", json={"title": "T", "questions": [{**qs[0], "correct_answer": ""}]}).status_code == 422

        files = [("files", (f"s{i}.pdf", f"%PDF sheet {i}".encode(), "application/pdf")) for i in range(12)]
        assert c.post(f"/api/exams/{exam_id}/sheets", files=files).json() == {"queued": 12}

        status = c.get(f"/api/exams/{exam_id}/status").json()
        assert len(status) == 12 and all(s["status"] in ("done", "needs_review") for s in status)

        res = c.get(f"/api/exams/{exam_id}/results").json()
        assert res["exam"]["total_marks"] == 40
        for r in res["rows"]:
            assert 0 <= r["total"] <= 40 and len(r["answers"]) == 20
        un = c.get(f"/api/exams/{exam_id}/unreadable").json()
        assert len(res["rows"]) + len(un) == 12

        if un:
            assert c.post(f"/api/exams/submissions/{un[0]['id']}/manual", json={"marks": 33}).json()["ok"] is True
            assert c.post(f"/api/exams/submissions/{un[0]['id']}/manual", json={"marks": 99}).status_code == 422
        x = c.get(f"/api/exams/{exam_id}/export")
        assert x.status_code == 200 and x.content[:2] == b"PK"
        d = c.get("/api/dashboard").json()
        assert d["exams"] == 1 and d["sheets_checked"] >= 1
        assert c.get("/api/config").json()["llm_ready"] is True
        assert c.post("/api/exams/parse-paper", files={"file": ("x.exe", b"x")}).status_code == 400


def test_answer_key_table_parser():
    from app.routes.exams import key_from_table
    rows = [("Q", "Answer"), ("Q1", "A"), ("2", "B"), ("Q3", 42.0), ("4", "3/4"), ("noise", "x"), ("5", None)]
    assert key_from_table(rows) == {"1": "A", "2": "B", "3": "42", "4": "3/4"}


def test_parse_key_reads_the_key_locally_before_asking_the_llm(monkeypatch):
    from app.services import answer_key_reader

    monkeypatch.setattr(answer_key_reader, "read_with_tesseract", lambda data, suffix: ("Q1: A\n2. 42\nQ3 - C", None))
    monkeypatch.setattr(answer_key_reader, "read_with_ollama",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("the LLM must not be called")))
    with TestClient(app) as client:
        response = client.post("/api/exams/parse-key", files={"file": ("key.pdf", b"%PDF-fake", "application/pdf")})
    assert response.status_code == 200
    assert response.json()["answers"] == {"1": "A", "2": "42", "3": "C"}
    assert response.json()["source"] == "tesseract"


def test_parse_key_reports_a_key_nobody_could_read(monkeypatch):
    from app.services import answer_key_reader

    monkeypatch.setattr(answer_key_reader, "read_with_tesseract", lambda data, suffix: ("", None))

    def fail(data, suffix):
        raise answer_key_reader.AnswerKeyReadError("model unavailable")
    monkeypatch.setattr(answer_key_reader, "read_with_ollama", fail)
    with TestClient(app) as client:
        response = client.post("/api/exams/parse-key", files={"file": ("key.pdf", b"%PDF-fake", "application/pdf")})
    assert response.status_code == 422
