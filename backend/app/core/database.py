
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

url = make_url(settings.DATABASE_URL)

is_sqlite = url.get_backend_name() == "sqlite"

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if is_sqlite else {},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Initialize database tables."""

    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db():
    """Provide a database session."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
