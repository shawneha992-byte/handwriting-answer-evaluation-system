"""Reference material for an exam (the 'knowledge base') and a no-scan preview of RAG grading."""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.access import optional_user
from app.core.config import settings
from app.core.database import get_db
from app.models import User
from app.rag import grader
from app.rag.embeddings import EmbeddingError, get_embedder
from app.rag.ingest import KNOWLEDGE_TYPES, IngestError, ingest_document, reindex_existing_chunks
from app.rag.models import KnowledgeChunk, KnowledgeDoc
from app.rag.retrieve import KnowledgeIndex, RagError
from app.routes.exams import get_exam, read_upload

router = APIRouter(prefix="/exams", tags=["knowledge"])


@router.post("/{exam_id}/knowledge")
def upload_knowledge(exam_id: int, files: list[UploadFile] = File(...), db: Session = Depends(get_db),
                     user: User | None = Depends(optional_user)):
    exam = get_exam(db, exam_id, user)
    try:
        embedder = get_embedder()
        docs = []
        for f in files:
            data, suffix = read_upload(f, KNOWLEDGE_TYPES)
            docs.append(ingest_document(db, exam.id, f.filename or "file", data, suffix, embedder))
    except IngestError as exc:
        raise HTTPException(422, str(exc))
    except EmbeddingError as exc:
        raise HTTPException(502, str(exc))
    return [{"id": d.id, "filename": d.filename, "chunks": d.n_chunks} for d in docs]



@router.post("/{exam_id}/knowledge/reindex")
def reindex_knowledge(exam_id: int, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    """Backfill pgvector embeddings for knowledge chunks created before the pgvector migration."""
    get_exam(db, exam_id, user)
    try:
        count = reindex_existing_chunks(db, exam_id, get_embedder())
    except EmbeddingError as exc:
        raise HTTPException(502, str(exc))
    return {"updated_chunks": count, "embedding_model": settings.EMBED_MODEL}


@router.get("/{exam_id}/knowledge")
def list_knowledge(exam_id: int, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    get_exam(db, exam_id, user)
    docs = db.query(KnowledgeDoc).filter(KnowledgeDoc.exam_id == exam_id).order_by(KnowledgeDoc.id).all()
    return [{"id": d.id, "filename": d.filename, "chunks": d.n_chunks} for d in docs]


@router.delete("/{exam_id}/knowledge/{doc_id}")
def delete_knowledge(exam_id: int, doc_id: int, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    get_exam(db, exam_id, user)
    doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id, KnowledgeDoc.exam_id == exam_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    db.query(KnowledgeChunk).filter(KnowledgeChunk.doc_id == doc.id).delete(synchronize_session=False)
    db.delete(doc)
    db.commit()
    return {"ok": True}


class PreviewIn(BaseModel):
    question_number: int
    answer_text: str


@router.post("/{exam_id}/grade-preview")
def grade_preview(exam_id: int, body: PreviewIn, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    """Grade typed text for one descriptive question, without scanning anything. Shows what was retrieved and why
    the marks were given, which is the quickest way to tune chunking, the rubric and RAG_MIN_SCORE."""
    exam = get_exam(db, exam_id, user)
    question = next((q for q in exam.questions if q.number == body.question_number), None)
    if question is None or question.type != "Descriptive":
        raise HTTPException(404, "That question does not exist or is not descriptive.")
    payload = grader.question_payload(question)
    try:
        index = KnowledgeIndex.load(db, exam.id, get_embedder())
        hits = index.search_many(grader.retrieval_queries(payload), settings.RAG_TOP_K) if index else []
    except (RagError, EmbeddingError) as exc:
        raise HTTPException(502, str(exc))
    result = grader.evaluate(payload, body.answer_text, hits)
    return {**result, "retrieved": [{"chunk_id": h.chunk_id, "source": h.source, "page": h.page,
                                     "score": round(h.vector_score, 3), "text": h.text[:400]} for h in hits]}
