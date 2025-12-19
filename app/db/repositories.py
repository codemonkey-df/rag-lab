"""
Database repository layer for CRUD operations
"""
from sqlmodel import Session, select
from uuid import UUID
from app.db.models import Document, QueryResult, Session as SessionModel


class DocumentRepository:
    """Repository for document operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, document: Document) -> Document:
        """Create a new document"""
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document
    
    def get_by_id(self, doc_id: UUID) -> Document | None:
        """Get document by ID"""
        return self.session.get(Document, doc_id)
    
    def list_by_session(self, session_id: UUID) -> list[Document]:
        """List all documents in a session"""
        statement = select(Document).where(Document.session_id == session_id)
        return list(self.session.exec(statement))
    
    def list_all(self) -> list[Document]:
        """List all documents"""
        statement = select(Document)
        return list(self.session.exec(statement))
    
    def update(self, document: Document) -> Document:
        """Update an existing document"""
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document
    
    def delete(self, doc_id: UUID) -> bool:
        """Delete a document"""
        document = self.get_by_id(doc_id)
        if document:
            self.session.delete(document)
            self.session.commit()
            return True
        return False


class QueryResultRepository:
    """Repository for query result operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, result: QueryResult) -> QueryResult:
        """Create a new query result"""
        self.session.add(result)
        self.session.commit()
        self.session.refresh(result)
        return result
    
    def get_by_id(self, result_id: UUID) -> QueryResult | None:
        """Get query result by ID"""
        return self.session.get(QueryResult, result_id)
    
    def list_by_session(self, session_id: UUID) -> list[QueryResult]:
        """List all query results in a session"""
        statement = select(QueryResult).where(QueryResult.session_id == session_id)
        return list(self.session.exec(statement))


class SessionRepository:
    """Repository for session operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, session_model: SessionModel) -> SessionModel:
        """Create a new session"""
        self.session.add(session_model)
        self.session.commit()
        self.session.refresh(session_model)
        return session_model
    
    def get_by_id(self, session_id: UUID) -> SessionModel | None:
        """Get session by ID"""
        return self.session.get(SessionModel, session_id)
