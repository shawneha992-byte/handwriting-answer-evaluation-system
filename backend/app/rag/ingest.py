"""Extract text from teacher-supplied reference material, chunk it, embed it and store it."""
import hashlib
from io import BytesIO

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.rag.chunking import chunk_pages
from app.rag.models import KnowledgeChunk, KnowledgeDoc
from app.services import question_paper_reader as reader

KNOWLEDGE_TYPES = {".pdf", ".txt", ".md", ".docx", ".jpg", ".jpeg", ".png"}


class IngestError(Exception):
    """The file could not be turned into searchable text."""


def extract_pages(data: bytes, suffix: str) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. Digital PDFs are copied exactly; scanned pages and photos use Tesseract if installed."""
    suffix = suffix.lower()
    pages: list[tuple[int, str]] = []
    if suffix in {".txt", ".md"}:
        pages.append((1, reader.clean_text(data.decode("utf-8-sig", errors="replace"))))
    elif suffix == ".docx":
        try:
            from docx import Document
            document = Document(BytesIO(data))
            text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
            # Include table contents because textbooks/answer keys often use tables.
            for table in document.tables:
                for row in table.rows:
                    text += "\n" + " | ".join(cell.text.strip() for cell in row.cells)
            pages.append((1, reader.clean_text(text)))
        except Exception as exc:
            raise IngestError(f"Could not read DOCX reference material: {exc}") from exc
    elif suffix == ".pdf":
        if reader.fitz is None:
            raise IngestError("PyMuPDF is not installed, so PDFs cannot be read.")
        try:
            doc = reader.fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise IngestError(f"Could not open this PDF: {exc}") from exc
        try:
            for number, page in enumerate(doc, start=1):
                text = reader.clean_text(page.get_text("text"))
                if len(text) < settings.MIN_PAGE_TEXT_CHARS and reader.ocr_available():
                    pix = page.get_pixmap(matrix=reader.fitz.Matrix(settings.OCR_ZOOM, settings.OCR_ZOOM), alpha=False)
                    text = reader._ocr(reader.Image.open(BytesIO(pix.tobytes("png"))))
                pages.append((number, text))
        finally:
            doc.close()
    elif suffix in {".jpg", ".jpeg", ".png"}:
        if not reader.ocr_available():
            raise IngestError("Reading images needs Tesseract OCR, which is not installed.")
        pages.append((1, reader._ocr(reader.Image.open(BytesIO(data)))))
    else:
        raise IngestError(f"Unsupported file type: {suffix}")
    pages = [(n, t) for n, t in pages if t and t.strip()]
    if not pages:
        raise IngestError("No readable text was found. A scanned file needs Tesseract OCR installed.")
    return pages


def ingest_document(db: Session, exam_id: int, filename: str, data: bytes, suffix: str, embedder) -> KnowledgeDoc:
    file_hash = hashlib.sha256(data).hexdigest()
    existing = db.query(KnowledgeDoc).filter(KnowledgeDoc.exam_id == exam_id, KnowledgeDoc.file_hash == file_hash).first()
    if existing:
        return existing  # the same file is never embedded twice for one exam
    chunks = chunk_pages(extract_pages(data, suffix), settings.RAG_CHUNK_CHARS, settings.RAG_CHUNK_OVERLAP)
    if not chunks:
        raise IngestError("The file produced no text chunks.")
    vectors = embedder.embed([c["text"] for c in chunks])
    doc = KnowledgeDoc(exam_id=exam_id, filename=filename[:255], file_hash=file_hash, n_chunks=len(chunks))
    db.add(doc)
    db.flush()
    db.add_all(
        KnowledgeChunk(doc_id=doc.id, exam_id=exam_id, page=c["page"], idx=c["idx"], text=c["text"],
                       embedding=np.asarray(vectors[i], dtype=np.float32).tobytes(),
                       embedding_vector=np.asarray(vectors[i], dtype=np.float32).tolist(),
                       embed_model=embedder.name, dim=int(vectors.shape[1]))
        for i, c in enumerate(chunks)
    )
    db.commit()
    return doc


def reindex_existing_chunks(db: Session, exam_id: int, embedder) -> int:
    """Create/update pgvector embeddings for chunks ingested by the earlier binary-vector version."""
    rows = db.query(KnowledgeChunk).filter(KnowledgeChunk.exam_id == exam_id).order_by(KnowledgeChunk.id).all()
    if not rows:
        return 0
    vectors = embedder.embed([row.text for row in rows])
    for row, vector in zip(rows, vectors):
        values = np.asarray(vector, dtype=np.float32)
        row.embedding = values.tobytes()
        row.embedding_vector = values.tolist()
        row.embed_model = embedder.name
        row.dim = int(values.shape[0])
    db.commit()
    return len(rows)
