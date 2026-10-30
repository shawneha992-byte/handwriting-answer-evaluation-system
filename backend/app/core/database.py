from sqlalchemy import create_engine, text
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
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


def ensure_database() -> None:
    """Create the MySQL database if it does not exist yet."""
    if url.get_backend_name() in ("mysql", "mariadb") and url.database:
        boot = create_engine(url.set(database=None))
        with boot.begin() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{url.database}` CHARACTER SET utf8mb4"))
        boot.dispose()


def init_db() -> None:
    from app import models  # noqa: F401  (register tables)

    ensure_database()
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
