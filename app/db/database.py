"""
Database engine and session management
"""

from functools import lru_cache

from sqlmodel import Session, SQLModel, create_engine, text

from app.core.config import get_settings

settings = get_settings()


@lru_cache()
def get_engine():
    """Get SQLite database engine"""
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},  # SQLite specific
        echo=False,  # Set to True for SQL logging
    )
    return engine


def migrate_database():
    """Migrate database schema if needed."""
    engine = get_engine()

    # Try to add error column - SQLite will raise an error if it already exists
    # This avoids needing SQLAlchemy's inspect module
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE document ADD COLUMN error TEXT"))
            conn.commit()
    except Exception:
        # Column already exists or table doesn't exist yet (will be created by create_all)
        # Ignore the error - create_all will handle table creation
        pass


def create_db_and_tables():
    """Create database tables and run migrations."""
    engine = get_engine()
    migrate_database()  # Run migrations first
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get database session (dependency for FastAPI)"""
    with Session(get_engine()) as session:
        yield session
