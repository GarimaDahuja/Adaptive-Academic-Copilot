"""
Gemini RAG (Retrieval-Augmented Generation) Service.
Handles 768-dimensional vector embedding generation, cosine similarity ranking,
and grounded contextual academic Q&A with verifiable source citations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from flask import current_app

from backend.app.models.document import DocumentChunk


class GeminiRAGService:
    """
    RAG service powered by the Google Gemini API.
    Provides semantic embeddings, nearest-neighbor chunk retrieval,
    and grounded answers with citations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-004",
        generation_model: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or (
            current_app.config.get("GEMINI_API_KEY", "") if current_app else ""
        )
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self._client = None

        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def is_live_api(self) -> bool:
        """Returns True if a live Gemini API client is initialized."""
        return self._client is not None

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a 768-dimensional normalized embedding vector for text.
        Falls back to a deterministic normalized unit vector when offline/in tests.
        """
        if not text or not text.strip():
            return [0.0] * 768

        if self.is_live_api:
            try:
                response = self._client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                )
                if response and response.embeddings:
                    return list(response.embeddings[0].values)
            except Exception as e:
                # Fall through to deterministic fallback if API fails
                if current_app and current_app.debug:
                    current_app.logger.warning(
                        f"Gemini API embed call failed ({e}), using fallback."
                    )

        # Deterministic token-accumulated 768-dim vector normalized to unit length
        # When two texts share words, their sum of word vectors will exhibit high positive cosine similarity
        words = [w.strip().lower() for w in text.split() if w.strip()]
        if not words:
            words = [text.strip().lower()]

        combined_vec = np.zeros(768, dtype=np.float32)
        for word in words:
            seed = abs(hash(word)) % (2**31)
            word_rng = np.random.default_rng(seed)
            combined_vec += word_rng.standard_normal(768).astype(np.float32)

        norm = np.linalg.norm(combined_vec)
        if norm > 0:
            combined_vec = combined_vec / norm
        return combined_vec.tolist()

    @staticmethod
    def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two numeric vectors in [-1.0, 1.0]."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def rank_chunks(
        self,
        query_embedding: List[float],
        candidate_chunks: List[DocumentChunk],
        top_k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Rank candidate document chunks by cosine similarity to the query embedding.
        Returns top_k (chunk, score) tuples ordered by highest similarity.
        """
        scored_chunks: List[Tuple[DocumentChunk, float]] = []

        for chunk in candidate_chunks:
            if chunk.embedding is not None:
                sim = self.compute_cosine_similarity(query_embedding, chunk.embedding)
                scored_chunks.append((chunk, sim))

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda item: item[1], reverse=True)
        return scored_chunks[:top_k]

    def generate_grounded_answer(
        self,
        question: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
    ) -> Dict[str, Any]:
        """
        Generate an academic answer grounded strictly in the retrieved document chunks.
        Extracts source citations pointing to specific documents and page numbers.
        """
        if not retrieved_chunks:
            return {
                "answer": (
                    "No relevant academic materials were found in the database "
                    "to answer this question. Please upload relevant course notes or PYQs."
                ),
                "citations": [],
                "sources_used": [],
                "confidence_score": 0.0,
            }

        # Build context prompt
        context_blocks = []
        citations_list = []

        for rank, (chunk, score) in enumerate(retrieved_chunks, 1):
            doc_title = chunk.document.title if chunk.document else "Unknown Document"
            page_num = chunk.page_number if chunk.page_number else "N/A"

            context_blocks.append(
                f"[Source {rank}]\n"
                f"Document: {doc_title}\n"
                f"Page: {page_num}\n"
                f"Content: {chunk.content}\n"
            )

            citations_list.append({
                "source_id": rank,
                "document_id": chunk.document_id,
                "document_title": doc_title,
                "page_number": chunk.page_number,
                "chunk_id": chunk.id,
                "similarity_score": round(score, 4),
            })

        joined_context = "\n---\n".join(context_blocks)
        avg_confidence = float(np.mean([score for _, score in retrieved_chunks]))

        system_instruction = (
            "You are an expert Academic Copilot assisting university students. "
            "Your task is to answer the student's question accurately, clearly, and concisely "
            "based ONLY on the provided academic sources. "
            "Every key fact or definition MUST cite the source using the format: "
            "[Document: {title}, Page: {page}]. "
            "If the provided context does not contain enough information to answer, explicitly state: "
            "'The provided course materials do not contain sufficient detail to fully answer this question.'"
        )

        user_prompt = (
            f"Academic Context Materials:\n{joined_context}\n\n"
            f"Student Question: {question}\n\n"
            f"Please provide a structured, well-explained answer with citations:"
        )

        if self.is_live_api:
            try:
                response = self._client.models.generate_content(
                    model=self.generation_model,
                    contents=user_prompt,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": 0.2,
                    },
                )
                answer_text = response.text
            except Exception as e:
                # If API call fails, format structured answer from retrieved chunks
                answer_text = self._format_offline_grounded_answer(
                    question, retrieved_chunks
                )
        else:
            answer_text = self._format_offline_grounded_answer(
                question, retrieved_chunks
            )

        return {
            "answer": answer_text,
            "citations": citations_list,
            "sources_used": [chunk.to_dict() for chunk, _ in retrieved_chunks],
            "confidence_score": round(avg_confidence, 4),
            "model_used": (
                self.generation_model if self.is_live_api else "offline-rag-engine"
            ),
        }

    def _format_offline_grounded_answer(
        self,
        question: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
    ) -> str:
        """Format an informative grounded answer from chunks when offline/testing."""
        top_chunk, _ = retrieved_chunks[0]
        doc_title = top_chunk.document.title if top_chunk.document else "Uploaded Material"
        page = top_chunk.page_number or 1

        answer_lines = [
            f"Based on your academic materials for **{doc_title}**:",
            "",
            f"> {top_chunk.content}",
            "",
            f"**Citation**: [Document: {doc_title}, Page: {page}]",
        ]
        return "\n".join(answer_lines)
