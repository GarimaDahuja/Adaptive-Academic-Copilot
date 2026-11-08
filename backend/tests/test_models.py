"""
Automated unit tests for Document, DocumentChunk, and Comment database models.
"""

import pytest
from backend.app.models.document import Document, DocumentChunk, Comment


def test_create_document_defaults(db_session):
    """Verify that a document is created with correct default values."""
    doc = Document(
        title="Data Structures End-Sem 2024",
        filename="ds_endsem_2024.pdf",
        file_type="pyq",
        subject="Data Structures",
        semester=3,
        uploader_id="student_123",
        file_size_bytes=1048576,
    )
    db_session.add(doc)
    db_session.commit()

    assert doc.id is not None
    assert doc.verification_status == "pending"
    assert doc.upvotes == 0
    assert doc.downvotes == 0
    assert doc.net_score == 0
    assert doc.created_at is not None

    data = doc.to_dict()
    assert data["title"] == "Data Structures End-Sem 2024"
    assert data["file_type"] == "pyq"
    assert data["verification_status"] == "pending"
    assert data["net_score"] == 0
    assert data["chunk_count"] == 0


def test_document_voting_and_net_score(db_session):
    """Verify upvotes, downvotes, and net score calculations."""
    doc = Document(
        title="Operating Systems Lab Manual",
        filename="os_lab_manual.pdf",
        file_type="lab_manual",
        subject="Operating Systems",
        semester=4,
    )
    db_session.add(doc)
    db_session.commit()

    doc.upvote()
    doc.upvote()
    doc.downvote()
    db_session.commit()

    assert doc.upvotes == 2
    assert doc.downvotes == 1
    assert doc.net_score == 1


def test_document_verification_transitions(db_session):
    """Verify status updates and rejection of invalid statuses."""
    doc = Document(
        title="Discrete Mathematics Notes",
        filename="dm_notes.pdf",
        file_type="notes",
        subject="Mathematics",
        semester=2,
    )
    db_session.add(doc)
    db_session.commit()

    # Valid status transition
    doc.set_verification("verified")
    db_session.commit()
    assert doc.verification_status == "verified"

    doc.set_verification("rejected")
    db_session.commit()
    assert doc.verification_status == "rejected"

    # Invalid status should raise ValueError
    with pytest.raises(ValueError):
        doc.set_verification("invalid_status_xyz")


def test_document_chunks_and_cascade_delete(db_session):
    """Verify chunk association and cascade deletion when parent document is deleted."""
    doc = Document(
        title="Algorithms Syllabus and Lecture 1",
        filename="algo_lec1.pdf",
        file_type="notes",
        subject="Algorithms",
        semester=3,
    )
    db_session.add(doc)
    db_session.commit()

    # Dummy 768-dim embedding (matching Gemini text-embedding-004)
    dummy_embedding = [0.01 * i for i in range(768)]

    chunk1 = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="Introduction to Asymptotic Notation: Big-O, Big-Omega, Big-Theta.",
        page_number=1,
        embedding=dummy_embedding,
    )
    chunk2 = DocumentChunk(
        document_id=doc.id,
        chunk_index=1,
        content="Divide and Conquer paradigm: MergeSort recurrence analysis.",
        page_number=2,
        embedding=dummy_embedding,
    )
    db_session.add_all([chunk1, chunk2])
    db_session.commit()

    assert len(doc.chunks) == 2
    assert doc.to_dict()["chunk_count"] == 2

    chunk_data = chunk1.to_dict()
    assert chunk_data["chunk_index"] == 0
    assert chunk_data["has_embedding"] is True
    assert "Asymptotic Notation" in chunk_data["content"]

    # Deleting parent document should cascade-delete all chunks
    db_session.delete(doc)
    db_session.commit()

    remaining_chunks = DocumentChunk.query.filter_by(document_id=doc.id).all()
    assert len(remaining_chunks) == 0


def test_document_comments(db_session):
    """Verify user comments can be attached to academic documents."""
    doc = Document(
        title="Computer Networks PYQ 2023",
        filename="cn_pyq_2023.pdf",
        file_type="pyq",
        subject="Computer Networks",
        semester=5,
    )
    db_session.add(doc)
    db_session.commit()

    comment = Comment(
        document_id=doc.id,
        user_id="student_456",
        comment_text="Question 3 has an erratum in the subnet mask given.",
    )
    db_session.add(comment)
    db_session.commit()

    assert len(doc.comments) == 1
    comment_data = comment.to_dict()
    assert comment_data["user_id"] == "student_456"
    assert "Question 3" in comment_data["comment_text"]
