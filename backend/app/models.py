from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Exam(Base):
    __tablename__ = "exams"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    total_marks: Mapped[float] = mapped_column(Float, default=0)
    negative_marking: Mapped[float] = mapped_column(Float, default=0)  # e.g. -0.5 per wrong answer
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # the teacher who created it
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    questions: Mapped[list["Question"]] = relationship(back_populates="exam", order_by="Question.number", cascade="all, delete-orphan")
    submissions: Mapped[list["Submission"]] = relationship(back_populates="exam", cascade="all, delete-orphan")


class Question(Base):
    """One question plus its answer key and marks (the marking scheme)."""
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"))
    number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, default="")
    type: Mapped[str] = mapped_column(String(30), default="MCQ")  # auto-detected question type
    marks: Mapped[float] = mapped_column(Float, default=1)
    correct_answer: Mapped[str] = mapped_column(Text, default="")
    rubric_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # Descriptive: [{"criterion": "...", "marks": 2}]
    model_answer: Mapped[str] = mapped_column(Text, default="")  # Descriptive: optional teacher model answer
    exam: Mapped[Exam] = relationship(back_populates="questions")


class Submission(Base):
    """One uploaded answer sheet (one student)."""
    __tablename__ = "submissions"
    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"))
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    roll_no: Mapped[str | None] = mapped_column(String(50), nullable=True)
    student_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued | reading | done | needs_review
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    manual: Mapped[bool] = mapped_column(Boolean, default=False)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_detail: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    exam: Mapped[Exam] = relationship(back_populates="submissions")
    answers: Mapped[list["Answer"]] = relationship(back_populates="submission", order_by="Answer.question_number", cascade="all, delete-orphan")


class Answer(Base):
    __tablename__ = "answers"
    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"))
    question_number: Mapped[int] = mapped_column(Integer)
    student_answer: Mapped[str] = mapped_column(Text, default="")
    correct_answer: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[str] = mapped_column(String(20))  # correct | partial | wrong | unanswered | unreadable
    marks: Mapped[float] = mapped_column(Float, default=0)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)  # Descriptive: per-criterion marks and reasons
    citations_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # Descriptive: sources behind the marks
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # best cosine match for the question
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)  # Descriptive: AI grade the teacher should check
    submission: Mapped[Submission] = relationship(back_populates="answers")


class PaperCache(Base):
    """Question papers already read, keyed by the SHA-256 of the uploaded file, so the same paper is never processed twice."""
    __tablename__ = "paper_cache"
    file_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    result_json: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(30), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class User(Base):
    """A teacher account. Each Exam has an owner_id pointing here, and the API only shows a teacher their own exams."""
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    institution: Mapped[str | None] = mapped_column(String(150), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# Imported last so the reference-material tables are created together with the rest.
from app.rag import models as _rag_models  # noqa: E402,F401
