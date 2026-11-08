"""
PDF text extraction and semantic chunking service.
Extracts page-by-page text from academic PDFs and creates overlapping text chunks for RAG.
"""

import io
import re
from typing import Any, BinaryIO, Dict, List, Union
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not text:
        return ""
    # Replace multiple whitespace/newlines with a single space or clean newline
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    page_number: int,
    start_chunk_index: int = 0,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks while preserving sentence boundaries where possible.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    chunks: List[Dict[str, Any]] = []
    chunk_index = start_chunk_index
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If not at the end of the text, try to break at a sentence or word boundary
        if end < text_len:
            boundary = cleaned.rfind(". ", start, end)
            if boundary != -1 and boundary > start + (chunk_size // 2):
                end = boundary + 1  # Include the period
            else:
                space_boundary = cleaned.rfind(" ", start, end)
                if space_boundary != -1 and space_boundary > start + (chunk_size // 2):
                    end = space_boundary

        segment = cleaned[start:end].strip()
        if segment:
            chunks.append({
                "chunk_index": chunk_index,
                "content": segment,
                "page_number": page_number,
            })
            chunk_index += 1

        # Advance start by chunk_size - chunk_overlap
        next_start = end - chunk_overlap
        if next_start <= start:
            start = end
        else:
            start = next_start

    return chunks


def extract_text_and_chunks(
    pdf_source: Union[BinaryIO, bytes, str],
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> List[Dict[str, Any]]:
    """
    Extract text from an academic PDF file or byte stream and return overlapping chunks.

    Args:
        pdf_source: File-like binary stream, bytes, or file path to a PDF.
        chunk_size: Target character length per chunk.
        chunk_overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        List of dictionaries containing chunk_index, content, and page_number.
    """
    if isinstance(pdf_source, bytes):
        stream = io.BytesIO(pdf_source)
    elif isinstance(pdf_source, str):
        stream = open(pdf_source, "rb")
    else:
        stream = pdf_source

    reader = PdfReader(stream)
    all_chunks: List[Dict[str, Any]] = []
    current_chunk_idx = 0

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        page_chunks = chunk_text(
            text=page_text,
            page_number=page_num,
            start_chunk_index=current_chunk_idx,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(page_chunks)
        current_chunk_idx += len(page_chunks)

    return all_chunks

