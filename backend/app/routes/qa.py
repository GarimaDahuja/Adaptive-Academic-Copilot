"""
Contextual Q&A and Semantic Search REST API endpoints.
Powered by Gemini embeddings and grounded LLM synthesis.
"""

from typing import List, Optional
from flask import Blueprint, jsonify, request

from backend.app.database import db
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.gemini_service import GeminiRAGService

qa_bp = Blueprint("qa", __name__, url_prefix="/api/qa")


@qa_bp.route("/ask", methods=["POST"])
def ask_question():
    """
    Semantic retrieval and grounded Q&A with source citations.

    JSON payload:
    {
        "question": "What is the difference between mutex and semaphore?",
        "subject": "Operating Systems",  (optional)
        "document_ids": [1, 2],           (optional)
        "top_k": 4                       (optional, default 4)
    }
    """
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    subject = data.get("subject", None)
    document_ids = data.get("document_ids", None)
    top_k = int(data.get("top_k", 4))

    if not question:
        return jsonify({
            "error": "Bad Request",
            "message": "Field 'question' is required and cannot be empty.",
        }), 400

    if top_k < 1 or top_k > 20:
        return jsonify({
            "error": "Bad Request",
            "message": "Field 'top_k' must be between 1 and 20.",
        }), 400

    # Build query for candidate document chunks
    query = DocumentChunk.query.join(Document)

    if subject and str(subject).strip():
        query = query.filter(Document.subject.ilike(f"%{str(subject).strip()}%"))

    if document_ids and isinstance(document_ids, list):
        query = query.filter(DocumentChunk.document_id.in_(document_ids))

    all_candidate_chunks = query.all()

    if not all_candidate_chunks:
        return jsonify({
            "question": question,
            "answer": (
                "No matching academic documents were found in the database. "
                "Please upload relevant course materials first."
            ),
            "citations": [],
            "sources_used": [],
            "confidence_score": 0.0,
            "model_used": "none",
        }), 200

    rag_service = GeminiRAGService()

    # Ensure candidate chunks have embeddings (embed on the fly if needed)
    needs_commit = False
    for chunk in all_candidate_chunks:
        if chunk.embedding is None:
            chunk.embedding = rag_service.generate_embedding(chunk.content)
            needs_commit = True

    if needs_commit:
        db.session.commit()

    # Generate query embedding
    query_embedding = rag_service.generate_embedding(question)

    # Rank chunks by semantic similarity
    ranked_chunks = rag_service.rank_chunks(
        query_embedding=query_embedding,
        candidate_chunks=all_candidate_chunks,
        top_k=top_k,
    )

    # Synthesize answer with citations
    result = rag_service.generate_grounded_answer(
        question=question,
        retrieved_chunks=ranked_chunks,
    )

    return jsonify({
        "question": question,
        "answer": result["answer"],
        "citations": result["citations"],
        "sources_used": result["sources_used"],
        "confidence_score": result["confidence_score"],
        "model_used": result["model_used"],
    }), 200


@qa_bp.route("/embed-document/<int:doc_id>", methods=["POST"])
def embed_document(doc_id: int):
    """
    Explicitly generate and store vector embeddings for all chunks of a document.
    """
    document = db.get_or_404(Document, doc_id)
    chunks = DocumentChunk.query.filter_by(document_id=doc_id).all()

    if not chunks:
        return jsonify({
            "message": "Document has no text chunks to embed.",
            "document_id": doc_id,
            "chunks_embedded": 0,
        }), 200

    rag_service = GeminiRAGService()
    embedded_count = 0

    for chunk in chunks:
        if chunk.embedding is None:
            chunk.embedding = rag_service.generate_embedding(chunk.content)
            embedded_count += 1

    if embedded_count > 0:
        db.session.commit()

    return jsonify({
        "message": f"Successfully generated embeddings for {embedded_count} chunks.",
        "document_id": doc_id,
        "total_chunks": len(chunks),
        "chunks_embedded": embedded_count,
    }), 200

