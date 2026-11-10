"""Tests for: exam ownership, safe Excel export, upload checks, stuck-sheet recovery, second reading, fractions."""
from datetime import datetime, timedelta
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.core.config import settings
from app.core.database import SessionLocal
from app.main import app
from app.models import Answer, Submission
from app.services import llm_service, processing
from app.services.answer_key_reader import extract_answers_from_text
from app.services.export import build_workbook

QUESTIONS = [{"number": 1, "text": "Pick one", "type": "MCQ", "marks": 2, "correct_answer": "B"},
             {"number": 2, "text": "2+2", "type": "Numeric", "marks": 2, "correct_answer": "4"}]


def register(client, email):
    r = client.post("/api/auth/register", json={"full_name": "Teacher Test", "email": email, "password": "S3cure-pass!"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def add_sheet(exam_id, status="queued", file_bytes=b"%PDF-sheet", name="s.pdf", tmp_dir=None):
    import tempfile
    from pathlib import Path
    path = Path(tmp_dir or tempfile.mkdtemp()) / name
    path.write_bytes(file_bytes)
    db = SessionLocal()
    sub = Submission(exam_id=exam_id, filename=name, stored_path=str(path), status=status)
    db.add(sub)
    db.commit()
    sid = sub.id
    db.close()
    return sid


# ------------------------------------------------------------------ 1. exams belong to their teacher
@pytest.fixture()
def locked(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    with TestClient(app) as client:
        yield client


def test_teachers_cannot_see_or_change_each_others_exams(locked):
    c = locked
    a, b = register(c, "a@example.com"), register(c, "b@example.com")
    exam_id = c.post("/api/exams", json={"title": "A's exam", "questions": QUESTIONS}, headers=a).json()["id"]
    sid = add_sheet(exam_id, status="needs_review")

    # the owner can use everything
    assert c.get(f"/api/exams/{exam_id}/results", headers=a).status_code == 200
    assert c.post(f"/api/exams/submissions/{sid}/manual", json={"marks": 1}, headers=a).status_code == 200

    # another teacher gets "not found" on every route, exactly like an exam that does not exist
    for method, path, kwargs in [
        ("get", f"/api/exams/{exam_id}/status", {}), ("get", f"/api/exams/{exam_id}/results", {}),
        ("get", f"/api/exams/{exam_id}/unreadable", {}), ("get", f"/api/exams/{exam_id}/export", {}),
        ("post", f"/api/exams/{exam_id}/retry-stuck", {}),
        ("post", f"/api/exams/{exam_id}/sheets", {"files": [("files", ("x.pdf", b"%PDF-1", "application/pdf"))]}),
        ("post", f"/api/exams/submissions/{sid}/manual", {"json": {"marks": 0}}),
    ]:
        r = getattr(c, method)(path, headers=b, **kwargs)
        assert r.status_code == 404, (path, r.status_code)
    assert c.get("/api/exams/999/results", headers=b).status_code == 404  # same answer as a missing exam

    assert c.get("/api/exams", headers=b).json() == []
    assert [e["title"] for e in c.get("/api/exams", headers=a).json()] == ["A's exam"]
    assert c.get("/api/dashboard", headers=b).json()["exams"] == 0
    assert c.get("/api/dashboard", headers=a).json()["exams"] == 1
    assert c.get("/api/dashboard", headers=b).json()["sheets_checked"] == 0


# ------------------------------------------------------------------ 3. spreadsheet cells are never formulas
def test_export_never_writes_formulas():
    exam = SimpleNamespace(total_marks=10)
    evil = '=HYPERLINK("http://evil.example","click")'
    rows = [{"roll_no": "1", "name": evil, "total": 5, "pct": 50, "correct": 1, "wrong": 0, "unanswered": 0,
             "answers": [{"number": 1, "student_answer": "=1+1", "correct_answer": "B", "result": "wrong", "marks": 0}]}]
    review = [{"roll_no": None, "filename": "=cmd", "detail": "+SUM(A1)", "reason": "@x"}]
    wb = load_workbook(BytesIO(build_workbook(exam, rows, review)))
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                assert cell.data_type != "f", (ws.title, cell.coordinate, cell.value)
    assert wb["Summary"]["B2"].value == evil          # the text is kept, just not executed
    assert wb["Summary"]["C2"].value == 5             # numbers stay numbers


# ------------------------------------------------------------------ 6c / 7. uploads: size and real content
def test_uploads_are_checked_by_content_and_size(monkeypatch):
    with TestClient(app) as c:
        post = lambda name, data: c.post("/api/exams/parse-paper", files={"file": (name, data, "application/octet-stream")})
        assert post("p.pdf", b"this is not a pdf at all").status_code == 400
        assert post("p.jpg", b"%PDF-1.4 renamed").status_code == 400
        assert post("p.png", b"\xff\xd8\xff jpeg bytes").status_code == 400
        assert post("p.pdf", b"").status_code == 400
        key = lambda name, data: c.post("/api/exams/parse-key", files={"file": (name, data, "application/octet-stream")})
        assert key("k.csv", b"\x00\x01\x02 binary").status_code == 400
        assert key("k.xlsx", b"not a zip").status_code == 400
        assert key("k.csv", b"Q,Answer\n1,A\n").status_code == 200
        monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
        assert post("big.pdf", b"%PDF-" + b"0" * (1024 * 1024)).status_code == 400
        assert post("ok.pdf", b"%PDF-demo").status_code == 200


# ------------------------------------------------------------------ 4. sheets stuck after a stopped server
def test_stuck_sheets_are_recovered():
    with TestClient(app) as c:
        exam_id = c.post("/api/exams", json={"title": "E", "questions": QUESTIONS}).json()["id"]
        queued = add_sheet(exam_id, "queued")
        stuck = add_sheet(exam_id, "reading")
        busy = add_sheet(exam_id, "reading")
        done = add_sheet(exam_id, "done")
        db = SessionLocal()
        old = datetime.utcnow() - timedelta(minutes=settings.STALE_MINUTES + 5)
        db.query(Submission).filter(Submission.id == stuck).update({"updated_at": old}, synchronize_session=False)
        db.commit()
        ids = processing.recover_stuck(db)
        states = {s.id: s.status for s in db.query(Submission)}
        db.close()
        assert sorted(ids) == sorted([queued, stuck])      # the long-stuck sheet is requeued, the fresh one is left alone
        assert states[stuck] == "queued" and states[busy] == "reading" and states[done] == "done"

        r = c.post(f"/api/exams/{exam_id}/retry-stuck")
        assert r.json() == {"requeued": 2}
        db = SessionLocal()
        after = {s.id: s.status for s in db.query(Submission)}
        db.close()
        assert after[queued] in ("done", "needs_review") and after[stuck] in ("done", "needs_review")
        assert after[busy] == "reading"


# ------------------------------------------------------------------ 6d. second reading catches disagreements
def _sheet(answers, roll="7"):
    return {"roll_no": roll, "roll_confidence": 0.95, "name": "Kid",
            "answers": [{"number": n, "answer": a, "confidence": 0.95} for n, a in answers]}


def run_second_read(monkeypatch, first, second):
    monkeypatch.setattr(settings, "SECOND_READ", True)
    calls = []

    def fake(data, mime, questions, max_dim=None):
        calls.append(max_dim)
        return second if max_dim else first
    monkeypatch.setattr(llm_service, "parse_sheet", fake)
    with TestClient(app) as c:
        exam_id = c.post("/api/exams", json={"title": "E", "questions": QUESTIONS}).json()["id"]
        sid = add_sheet(exam_id)
        processing.process_submission(sid)
    db = SessionLocal()
    sub = db.get(Submission, sid)
    answers = {a.question_number: a for a in db.query(Answer).filter(Answer.submission_id == sid)}
    out = (sub.status, sub.review_reason, sub.review_detail, {n: a.result for n, a in answers.items()}, calls)
    db.close()
    return out


def test_two_readings_that_agree_are_accepted(monkeypatch):
    status, reason, detail, results, calls = run_second_read(monkeypatch, _sheet([(1, "B"), (2, "4")]), _sheet([(1, "b"), (2, "4.0")]))
    assert status == "done" and results == {1: "correct", 2: "correct"} and calls == [None, settings.SECOND_READ_MAX_DIM]


def test_two_readings_that_disagree_go_to_review(monkeypatch):
    status, reason, detail, results, _ = run_second_read(monkeypatch, _sheet([(1, "B"), (2, "4")]), _sheet([(1, "D"), (2, "4")]))
    assert status == "needs_review" and "disagree" in reason and detail == "Q1"
    assert results == {1: "unreadable", 2: "correct"}


def test_roll_number_mismatch_between_readings_goes_to_review(monkeypatch):
    status, reason, detail, _, _ = run_second_read(monkeypatch, _sheet([(1, "B"), (2, "4")], roll="7"), _sheet([(1, "B"), (2, "4")], roll="1"))
    assert status == "needs_review" and "Roll number differs" in reason


def test_failed_second_reading_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(settings, "SECOND_READ", True)

    def fake(data, mime, questions, max_dim=None):
        if max_dim:
            raise llm_service.LLMError("ollama busy")
        return _sheet([(1, "B"), (2, "4")])
    monkeypatch.setattr(llm_service, "parse_sheet", fake)
    with TestClient(app) as c:
        exam_id = c.post("/api/exams", json={"title": "E", "questions": QUESTIONS}).json()["id"]
        sid = add_sheet(exam_id)
        processing.process_submission(sid)
    db = SessionLocal()
    sub = db.get(Submission, sid)
    assert sub.status == "needs_review" and "second reading failed" in sub.review_reason
    db.close()


# ------------------------------------------------------------------ 6e. fractions in an answer key
def test_fractions_survive_but_working_is_still_rejected():
    text = "1. 1/2\n2) -3/4\nQ3: 10/20 + 5\n4. 2*3\n5. 6/ x"
    assert extract_answers_from_text(text) == {"1": "1/2", "2": "-3/4"}
