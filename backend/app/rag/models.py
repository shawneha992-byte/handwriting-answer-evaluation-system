"""PostgreSQL-backed reference documents and pgvector embeddings.

The binary embedding column is retained for backwards compatibility and SQLite tests.
Production PostgreSQL retrieval uses embedding_vector (pgvector).
"""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.core.config import settings
from app.core.database import Base

_VECTOR_TYPE = Vector(settings.RAG_EMBED_DIM).with_variant(JSON(), "sqlite")


class KnowledgeDoc(Base):
    __tablename__ = "knowledge_docs"
    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_hash: Mapped[str] = mapped_column(String(64))
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("knowledge_docs.id", ondelete="CASCADE"), index=True)
    exam_id: Mapped[int] = mapped_column(Integer, index=True)
    page: Mapped[int] = mapped_column(Integer, default=1)
    idx: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    # Legacy serialized vector: used only for SQLite test compatibility and rollback safety.
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    # Real vector column in PostgreSQL; JSON fallback keeps unit tests on SQLite working.
    embedding_vector: Mapped[list | None] = mapped_column(_VECTOR_TYPE, nullable=True)
    embed_model: Mapped[str] = mapped_column(String(100))
    dim: Mapped[int] = mapped_column(Integer)
