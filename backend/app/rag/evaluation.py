"""Measure how close the AI grader is to a teacher. Run it on answers a teacher has already marked.

  python -m app.rag.run_eval --exam 3 --csv labeled.csv --k 6
CSV columns: question_number, answer_text, teacher_marks
"""
import numpy as np


def quadratic_weighted_kappa(a: list[int], b: list[int]) -> float | None:
    """Agreement between two integer ratings, 1 = identical, 0 = chance. The standard metric for automated scoring."""
    if not a or len(a) != len(b):
        return None
    low, high = min(a + b), max(a + b)
    k = high - low + 1
    if k < 2:
        return 1.0 if a == b else None
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[x - low, y - low] += 1
    expected = np.outer(observed.sum(1), observed.sum(0)) / len(a)
    weights = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    denom = (weights * expected).sum()
    return None if denom == 0 else float(1 - (weights * observed).sum() / denom)


def summarize(rows: list[dict], step: float = 0.25) -> dict:
    """rows: [{"teacher": float, "ai": float, "max": float, "needs_review": bool}]"""
    if not rows:
        return {"n": 0}
    teacher = np.array([r["teacher"] for r in rows], dtype=float)
    ai = np.array([r["ai"] for r in rows], dtype=float)
    full = np.array([r["max"] for r in rows], dtype=float)
    err = ai - teacher
    corr = float(np.corrcoef(teacher, ai)[0, 1]) if len(rows) > 1 and teacher.std() > 0 and ai.std() > 0 else None
    return {
        "n": len(rows),
        "mean_abs_error": round(float(np.abs(err).mean()), 3),
        "mean_abs_error_pct_of_max": round(float((np.abs(err) / np.where(full > 0, full, 1)).mean() * 100), 1),
        "bias_ai_minus_teacher": round(float(err.mean()), 3),
        "within_half_mark": round(float((np.abs(err) <= 0.5).mean()), 3),
        "exact": round(float((np.abs(err) < 1e-9).mean()), 3),
        "pearson": None if corr is None else round(corr, 3),
        "qwk": quadratic_weighted_kappa([int(round(t / step)) for t in teacher], [int(round(x / step)) for x in ai]),
        "sent_to_review": round(float(np.mean([r["needs_review"] for r in rows])), 3),
    }


def run(db, exam, rows: list[dict], k: int, embedder, chat=None) -> dict:
    """Grade each labelled row and compare with the teacher. k = chunks retrieved (0 = no retrieval)."""
    from app.rag import grader
    from app.rag.retrieve import KnowledgeIndex

    index = KnowledgeIndex.load(db, exam.id, embedder) if k > 0 else None
    questions = {q.number: q for q in exam.questions if q.type == "Descriptive"}
    scored = []
    for row in rows:
        q = questions.get(int(row["question_number"]))
        if q is None:
            continue
        payload = grader.question_payload(q)
        hits = index.search_many(grader.retrieval_queries(payload), k) if index else []
        out = grader.evaluate(payload, row["answer_text"], hits, chat)
        scored.append({"teacher": float(row["teacher_marks"]), "ai": out["marks"], "max": payload["marks"],
                       "needs_review": out["needs_review"]})
    return {"k": k, "summary": summarize(scored), "rows": scored}
