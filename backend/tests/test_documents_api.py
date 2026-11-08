"""
Automated unit and integration tests for Document REST API endpoints.
"""

import io
import pytest
from pypdf import PdfWriter

from backend.app.models.document import Document, DocumentChunk


def create_dummy_pdf_bytes() -> bytes:
    """Create a minimal valid PDF in memory for testing."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer.getvalue()


class TestDocumentUploadAPI:
    """Tests for POST /api/documents/upload."""

    def test_upload_valid_pdf(self, client):
        pdf_bytes = create_dummy_pdf_bytes()
        data = {
            "file": (io.BytesIO(pdf_bytes), "sample_pyq.pdf"),
            "title": "Algorithms Mid-Sem 2024",
            "subject": "Algorithms",
            "file_type": "pyq",
            "semester": "4",
            "uploader_id": "student_001",
        }
        response = client.post(
            "/api/documents/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 201
        res_data = response.get_json()
        assert "document" in res_data
        assert res_data["document"]["title"] == "Algorithms Mid-Sem 2024"
        assert res_data["document"]["verification_status"] == "pending"

    def test_upload_missing_file(self, client):
        response = client.post(
            "/api/documents/upload",
            data={"title": "Test Title", "subject": "Test Sub"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "No file part" in response.get_json()["message"]

    def test_upload_invalid_extension(self, client):
        data = {
            "file": (io.BytesIO(b"not a pdf"), "notes.txt"),
            "title": "Text Notes",
            "subject": "CS",
        }
        response = client.post(
            "/api/documents/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "Only PDF files are supported" in response.get_json()["message"]

    def test_upload_invalid_file_type(self, client):
        pdf_bytes = create_dummy_pdf_bytes()
        data = {
            "file": (io.BytesIO(pdf_bytes), "file.pdf"),
            "title": "Title",
            "subject": "Subject",
            "file_type": "unknown_type_xyz",
        }
        response = client.post(
            "/api/documents/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "Invalid file_type" in response.get_json()["message"]


class TestDocumentManagementAPI:
    """Tests for GET, voting, verification, and comments."""

    def test_list_and_filter_documents(self, client, db_session):
        doc1 = Document(
            title="Database Management Lab Manual",
            filename="db_lab.pdf",
            file_type="lab_manual",
            subject="Databases",
            semester=4,
        )
        doc2 = Document(
            title="Computer Networks PYQ 2023",
            filename="cn_pyq.pdf",
            file_type="pyq",
            subject="Networks",
            semester=5,
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        # List all
        res = client.get("/api/documents")
        assert res.status_code == 200
        assert res.get_json()["count"] >= 2

        # Filter by file_type
        res_filter = client.get("/api/documents?file_type=lab_manual")
        assert res_filter.status_code == 200
        data = res_filter.get_json()
        assert len(data["documents"]) == 1
        assert data["documents"][0]["file_type"] == "lab_manual"

    def test_get_document_by_id(self, client, db_session):
        doc = Document(
            title="Discrete Maths Notes",
            filename="dm.pdf",
            file_type="notes",
            subject="Mathematics",
        )
        db_session.add(doc)
        db_session.commit()

        res = client.get(f"/api/documents/{doc.id}")
        assert res.status_code == 200
        assert res.get_json()["document"]["title"] == "Discrete Maths Notes"

    def test_vote_document(self, client, db_session):
        doc = Document(
            title="OS Notes",
            filename="os.pdf",
            file_type="notes",
            subject="OS",
        )
        db_session.add(doc)
        db_session.commit()

        # Upvote
        res_up = client.post(
            f"/api/documents/{doc.id}/vote",
            json={"vote_type": "up"},
        )
        assert res_up.status_code == 200
        assert res_up.get_json()["upvotes"] == 1
        assert res_up.get_json()["net_score"] == 1

        # Downvote
        res_down = client.post(
            f"/api/documents/{doc.id}/vote",
            json={"vote_type": "down"},
        )
        assert res_down.status_code == 200
        assert res_down.get_json()["downvotes"] == 1
        assert res_down.get_json()["net_score"] == 0

    def test_verify_document(self, client, db_session):
        doc = Document(
            title="Compiler Design PYQ",
            filename="cd.pdf",
            file_type="pyq",
            subject="Compilers",
        )
        db_session.add(doc)
        db_session.commit()

        res = client.post(
            f"/api/documents/{doc.id}/verify",
            json={"status": "verified"},
        )
        assert res.status_code == 200
        assert res.get_json()["document"]["verification_status"] == "verified"

        # Invalid status returns 400
        res_invalid = client.post(
            f"/api/documents/{doc.id}/verify",
            json={"status": "bogus_status"},
        )
        assert res_invalid.status_code == 400

    def test_comments_endpoints(self, client, db_session):
        doc = Document(
            title="Python Lab Experiments",
            filename="py_lab.pdf",
            file_type="lab_manual",
            subject="Python",
        )
        db_session.add(doc)
        db_session.commit()

        # Add comment
        res_add = client.post(
            f"/api/documents/{doc.id}/comments",
            json={"user_id": "student_999", "comment_text": "Experiment 4 solution has a typo."},
        )
        assert res_add.status_code == 201
        assert "comment" in res_add.get_json()

        # List comments
        res_list = client.get(f"/api/documents/{doc.id}/comments")
        assert res_list.status_code == 200
        comments = res_list.get_json()["comments"]
        assert len(comments) == 1
        assert comments[0]["user_id"] == "student_999"

    def test_get_document_chunks(self, client, db_session):
        doc = Document(
            title="Database Systems Syllabus",
            filename="db_syllabus.pdf",
            file_type="syllabus",
            subject="DBMS",
        )
        db_session.add(doc)
        db_session.commit()

        chunk1 = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Unit 1: ER Modeling and Relational Algebra",
        )
        chunk2 = DocumentChunk(
            document_id=doc.id,
            chunk_index=1,
            page_number=2,
            content="Unit 2: SQL and Normalization up to BCNF",
        )
        db_session.add_all([chunk1, chunk2])
        db_session.commit()

        res = client.get(f"/api/documents/{doc.id}/chunks")
        assert res.status_code == 200
        data = res.get_json()
        assert data["document_id"] == doc.id
        assert data["count"] == 2
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["chunk_index"] == 0
        assert "ER Modeling" in data["chunks"][0]["content"]
        assert data["chunks"][1]["chunk_index"] == 1
        assert "Normalization" in data["chunks"][1]["content"]

