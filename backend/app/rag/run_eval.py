"""Command line wrapper:  python -m app.rag.run_eval --exam 3 --csv labeled.csv --k 6"""
import argparse
import csv
import json

from app.core.database import SessionLocal
from app.models import Exam
from app.rag import evaluation
from app.rag.embeddings import get_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AI marks with a teacher's marks on labelled answers.")
    parser.add_argument("--exam", type=int, required=True)
    parser.add_argument("--csv", required=True, help="columns: question_number, answer_text, teacher_marks")
    parser.add_argument("--k", type=int, default=6, help="chunks to retrieve; run with 0 to see what retrieval adds")
    args = parser.parse_args()
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    db = SessionLocal()
    try:
        exam = db.get(Exam, args.exam)
        if exam is None:
            raise SystemExit(f"Exam {args.exam} not found.")
        report = evaluation.run(db, exam, rows, args.k, get_embedder())
    finally:
        db.close()
    print(json.dumps({"k": report["k"], **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
