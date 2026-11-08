"""
Automated unit and integration tests for Gemini RAG Service and Q&A API.
"""

import pytest
import numpy as np

from backend.app.models.document import Document, DocumentChunk
from backend.app.services.gemini_service import GeminiRAGService


class TestGeminiRAGService:
    """Unit tests for the underlying RAG service algorithms."""

    def test_embedding_dimensions_and_normalization(self):
        service = GeminiRAGService()
        vec = service.generate_embedding("CPU Scheduling Algorithms")

        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(x, float) for x in vec)

        # Norm should be approximately 1.0 (unit vector)
        norm = np.linalg.norm(np.array(vec))
        assert abs(norm - 1.0) < 1e-4

    def test_cosine_similarity_computation(self):
        service = GeminiRAGService()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        # Identical vectors -> similarity = 1.0
        assert abs(service.compute_cosine_similarity(vec1, vec2) - 1.0) < 1e-5
        # Orthogonal vectors -> similarity = 0.0
        assert abs(service.compute_cosine_similarity(vec1, vec3) - 0.0) < 1e-5

    def test_ranking_chunks(self, db_session):
        service = GeminiRAGService()
        doc = Document(
            title="OS Lecture Notes",
            filename="os.pdf",
            file_type="notes",
            subject="OS",
        )
        db_session.add(doc)
        db_session.commit()

        chunk1 = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            content="Round Robin scheduling gives each process a time quantum.",
            page_number=5,
            embedding=service.generate_embedding("Round Robin time quantum scheduling"),
        )
        chunk2 = DocumentChunk(
            document_id=doc.id,
            chunk_index=1,
            content="Virtual Memory uses paging and translation lookaside buffers.",
            page_number=12,
            embedding=service.generate_embedding("Virtual Memory paging TLB"),
        )
        db_session.add_all([chunk1, chunk2])
        db_session.commit()

        query_emb = service.generate_embedding("Tell me about time slice and quantum in Round Robin")
        ranked = service.rank_chunks(query_emb, [chunk1, chunk2], top_k=1)

        assert len(ranked) == 1
        best_chunk, score = ranked[0]
        assert best_chunk.id == chunk1.id
        assert score > 0.0


class TestQAAPI:
    """Integration tests for POST /api/qa/ask and embedding endpoints."""

    def test_ask_empty_question_returns_400(self, client):
        res = client.post("/api/qa/ask", json={"question": ""})
        assert res.status_code == 400
        assert "cannot be empty" in res.get_json()["message"]

    def test_ask_invalid_top_k_returns_400(self, client):
        res = client.post("/api/qa/ask", json={"question": "What is Python?", "top_k": 999})
        assert res.status_code == 400
        assert "must be between 1 and 20" in res.get_json()["message"]

    def test_ask_no_documents_in_database(self, client):
        res = client.post("/api/qa/ask", json={"question": "What is a B-Tree?"})
        assert res.status_code == 200
        data = res.get_json()
        assert "No matching academic documents" in data["answer"]
        assert len(data["citations"]) == 0

    def test_ask_grounded_answer_with_citations(self, client, db_session):
        # Seed an academic document with chunks
        doc = Document(
            title="Operating Systems Concepts",
            filename="os_concepts.pdf",
            file_type="notes",
            subject="Operating Systems",
            semester=4,
        )
        db_session.add(doc)
        db_session.commit()

        service = GeminiRAGService()
        chunk1 = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            content="A semaphore is a synchronization variable that can only be accessed through wait() and signal().",
            page_number=42,
            embedding=service.generate_embedding("semaphore synchronization wait signal"),
        )
        chunk2 = DocumentChunk(
            document_id=doc.id,
            chunk_index=1,
            content="Deadlock requires four necessary conditions: mutual exclusion, hold and wait, no preemption, and circular wait.",
            page_number=55,
            embedding=service.generate_embedding("deadlock mutual exclusion circular wait"),
        )
        db_session.add_all([chunk1, chunk2])
        db_session.commit()

        # Ask question
        res = client.post(
            "/api/qa/ask",
            json={
                "question": "What is a semaphore and what operations are permitted on it?",
                "subject": "Operating Systems",
                "top_k": 2,
            },
        )
        assert res.status_code == 200
        data = res.get_json()

        assert "answer" in data
        assert "citations" in data
        assert len(data["citations"]) >= 1

        top_citation = data["citations"][0]
        assert top_citation["document_title"] == "Operating Systems Concepts"
        assert top_citation["page_number"] == 42
        assert "confidence_score" in data

    def test_embed_document_endpoint(self, client, db_session):
        doc = Document(
            title="Database Internals",
            filename="db.pdf",
            file_type="notes",
            subject="Databases",
        )
        db_session.add(doc)
        db_session.commit()

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            content="WAL (Write-Ahead Logging) ensures durability and atomicity in ACID.",
            page_number=10,
            embedding=None,  # Not yet embedded
        )
        db_session.add(chunk)
        db_session.commit()

        res = client.post(f"/api/qa/embed-document/{doc.id}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["chunks_embedded"] == 1

        # Verify chunk in DB now has an embedding
        updated_chunk = db_session.get(DocumentChunk, chunk.id)
        assert updated_chunk.embedding is not None

