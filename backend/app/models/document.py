"""
Document, DocumentChunk, and Comment models for academic materials.
Supports PDF metadata, community verification, and vector embeddings for RAG retrieval.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from sqlalchemy import Text, TypeDecorator
from backend.app.database import db


class VectorType(TypeDecorator):
    """
    Dialect-aware Vector type decorator.
    Uses pgvector's Vector(768) in PostgreSQL, and falls back to JSON Text in SQLite for testing.
    """

    impl = Text
    cache_ok = True

    def __init__(self, dim: int = 768, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector
                return dialect.type_descriptor(Vector(self.dim))
            except ImportError:
                return dialect.type_descriptor(Text())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        try:
            return json.loads(value)
        except Exception:
            return value


class Document(db.Model):
    """
    Academic material metadata model.
    Represents uploaded PDFs (PYQs, Lab Manuals, Notes, etc.).
    """

    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False, default="notes")
    subject = db.Column(db.String(100), nullable=False)
    semester = db.Column(db.Integer, nullable=True)
    uploader_id = db.Column(db.String(100), nullable=True)

    # Verification and reputation
    verification_status = db.Column(db.String(20), nullable=False, default="pending")
    upvotes = db.Column(db.Integer, nullable=False, default=0)
    downvotes = db.Column(db.Integer, nullable=False, default=0)
    file_size_bytes = db.Column(db.Integer, nullable=True)

    # Timestamps
    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    chunks = db.relationship(
        "DocumentChunk",
        backref="document",
        cascade="all, delete-orphan",
        lazy=True,
    )
    comments = db.relationship(
        "Comment",
        backref="document",
        cascade="all, delete-orphan",
        lazy=True,
    )

    @property
    def net_score(self) -> int:
        """Calculate net community reputation score (upvotes minus downvotes)."""
        return self.upvotes - self.downvotes

    def upvote(self) -> int:
        self.upvotes += 1
        return self.upvotes

    def downvote(self) -> int:
        self.downvotes += 1
        return self.downvotes

    def set_verification(self, status: str) -> None:
        valid_statuses = ("pending", "verified", "rejected")
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of: {valid_statuses}"
            )
        self.verification_status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "filename": self.filename,
            "file_type": self.file_type,
            "subject": self.subject,
            "semester": self.semester,
            "uploader_id": self.uploader_id,
            "verification_status": self.verification_status,
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
            "net_score": self.net_score,
            "file_size_bytes": self.file_size_bytes,
            "chunk_count": len(self.chunks),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DocumentChunk(db.Model):
    """
    Text chunks extracted from uploaded documents, paired with vector embeddings.
    Used for semantic retrieval (RAG) and citation matching.
    """

    __tablename__ = "document_chunks"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(
        db.Integer,
        db.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = db.Column(db.Integer, nullable=False)
    content = db.Column(db.Text, nullable=False)
    page_number = db.Column(db.Integer, nullable=True)

    # 768-dimensional vector embedding (matching Gemini text-embedding-004)
    embedding = db.Column(VectorType(dim=768), nullable=True)

    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "page_number": self.page_number,
            "has_embedding": self.embedding is not None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Comment(db.Model):
    """
    User discussion comment attached to an academic document.
    """

    __tablename__ = "comments"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(
        db.Integer,
        db.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = db.Column(db.String(100), nullable=False)
    comment_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "comment_text": self.comment_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
