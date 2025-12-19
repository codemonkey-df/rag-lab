"""
Database engine and session management
"""
from functools import lru_cache
from sqlmodel import SQLModel, create_engine, Session
from app.core.config import get_settings

settings = get_settings()


@lru_cache()
def get_engine():
    """Get SQLite database engine"""
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},  # SQLite specific
        echo=False  # Set to True for SQL logging
    )
    return engine


def create_db_and_tables():
    """Create database tables"""
    engine = get_engine()
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get database session (dependency for FastAPI)"""
    with Session(get_engine()) as session:
        yield session
