"""
Document management REST API endpoints.
Handles PDF uploads, chunking, verification, voting, and comments.
"""

from pathlib import Path
import time
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from backend.app.database import db
from backend.app.models.document import Document, DocumentChunk, Comment
from backend.app.services.pdf_service import extract_text_and_chunks

documents_bp = Blueprint("documents", __name__, url_prefix="/api/documents")

ALLOWED_FILE_TYPES = {"pyq", "lab_manual", "notes", "syllabus"}


def is_allowed_file(filename: str) -> bool:
    """Check if file has an allowed PDF extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config.get(
        "ALLOWED_EXTENSIONS", {"pdf"}
    )


@documents_bp.route("/upload", methods=["POST"])
def upload_document():
    """
    Upload an academic PDF with metadata, extract text into chunks, and store in DB.
    """
    if "file" not in request.files:
        return jsonify({"error": "Bad Request", "message": "No file part in request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Bad Request", "message": "No file selected for upload."}), 400

    if not is_allowed_file(file.filename):
        return jsonify({
            "error": "Bad Request",
            "message": "Only PDF files are supported (.pdf).",
        }), 400

    # Extract and validate form metadata
    title = request.form.get("title", "").strip()
    subject = request.form.get("subject", "").strip()
    file_type = request.form.get("file_type", "notes").strip().lower()
    semester_raw = request.form.get("semester", None)
    uploader_id = request.form.get("uploader_id", "anonymous").strip()

    if not title:
        return jsonify({"error": "Bad Request", "message": "Field 'title' is required."}), 400
    if not subject:
        return jsonify({"error": "Bad Request", "message": "Field 'subject' is required."}), 400
    if file_type not in ALLOWED_FILE_TYPES:
        return jsonify({
            "error": "Bad Request",
            "message": f"Invalid file_type '{file_type}'. Allowed types: {list(ALLOWED_FILE_TYPES)}",
        }), 400

    semester = None
    if semester_raw is not None and str(semester_raw).strip() != "":
        try:
            semester = int(semester_raw)
        except ValueError:
            return jsonify({
                "error": "Bad Request",
                "message": "Field 'semester' must be an integer.",
            }), 400

    # Read binary stream to inspect size and extract chunks
    file_bytes = file.read()
    file_size = len(file_bytes)

    # Save file to uploads directory
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_filename = secure_filename(file.filename)
    unique_filename = f"{int(time.time())}_{original_filename}"
    save_path = upload_dir / unique_filename

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    # Create Document record
    document = Document(
        title=title,
        filename=unique_filename,
        file_type=file_type,
        subject=subject,
        semester=semester,
        uploader_id=uploader_id,
        file_size_bytes=file_size,
    )
    db.session.add(document)
    db.session.flush()  # Flush to generate document.id

    # Parse PDF and create DocumentChunk records
    try:
        chunks_data = extract_text_and_chunks(file_bytes)
        for c in chunks_data:
            chunk_record = DocumentChunk(
                document_id=document.id,
                chunk_index=c["chunk_index"],
                content=c["content"],
                page_number=c["page_number"],
                embedding=None,  # Will be populated in Gemini RAG milestone
            )
            db.session.add(chunk_record)

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        # Clean up saved file if parsing completely fails
        if save_path.exists():
            save_path.unlink()
        return jsonify({
            "error": "Internal Server Error",
            "message": f"Failed to extract text from PDF: {str(e)}",
        }), 500

    return jsonify({
        "message": "Document uploaded and processed successfully.",
        "document": document.to_dict(),
    }), 201


@documents_bp.route("", methods=["GET"])
def list_documents():
    """
    List all uploaded academic documents with optional filtering.
    Filters: ?subject=...&file_type=...&semester=...&verification_status=...
    """
    query = Document.query

    subject = request.args.get("subject")
    if subject:
        query = query.filter(Document.subject.ilike(f"%{subject.strip()}%"))

    file_type = request.args.get("file_type")
    if file_type:
        query = query.filter(Document.file_type == file_type.strip().lower())

    semester = request.args.get("semester")
    if semester:
        try:
            query = query.filter(Document.semester == int(semester))
        except ValueError:
            pass

    verification_status = request.args.get("verification_status")
    if verification_status:
        query = query.filter(
            Document.verification_status == verification_status.strip().lower()
        )

    # Order by net score (highest reputation first) and recent upload
    documents = query.order_by(
        (Document.upvotes - Document.downvotes).desc(),
        Document.created_at.desc(),
    ).all()

    return jsonify({
        "count": len(documents),
        "documents": [doc.to_dict() for doc in documents],
    }), 200


@documents_bp.route("/<int:doc_id>", methods=["GET"])
def get_document(doc_id: int):
    """Retrieve metadata for a specific document."""
    document = db.get_or_404(Document, doc_id)
    return jsonify({"document": document.to_dict()}), 200


@documents_bp.route("/<int:doc_id>/chunks", methods=["GET"])
def get_document_chunks(doc_id: int):
    """Retrieve all parsed text chunks for a document."""
    document = db.get_or_404(Document, doc_id)
    chunks = (
        DocumentChunk.query.filter_by(document_id=doc_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )
    return jsonify({
        "document_id": doc_id,
        "count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }), 200


@documents_bp.route("/<int:doc_id>/vote", methods=["POST"])
def vote_document(doc_id: int):
    """
    Upvote or downvote a document to adjust contributor reputation.
    Request body: {"vote_type": "up" | "down"}
    """
    document = db.get_or_404(Document, doc_id)
    data = request.get_json() or {}
    vote_type = data.get("vote_type", "").lower()

    if vote_type == "up":
        document.upvote()
    elif vote_type == "down":
        document.downvote()
    else:
        return jsonify({
            "error": "Bad Request",
            "message": "Invalid vote_type. Must be 'up' or 'down'.",
        }), 400

    db.session.commit()
    return jsonify({
        "message": "Vote recorded successfully.",
        "upvotes": document.upvotes,
        "downvotes": document.downvotes,
        "net_score": document.net_score,
    }), 200


@documents_bp.route("/<int:doc_id>/verify", methods=["POST"])
def verify_document(doc_id: int):
    """
    Update document verification status (e.g., by peer reviewer or teacher).
    Request body: {"status": "verified" | "rejected" | "pending"}
    """
    document = db.get_or_404(Document, doc_id)
    data = request.get_json() or {}
    status = data.get("status", "").lower()

    try:
        document.set_verification(status)
        db.session.commit()
    except ValueError as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400

    return jsonify({
        "message": "Verification status updated successfully.",
        "document": document.to_dict(),
    }), 200


@documents_bp.route("/<int:doc_id>/comments", methods=["POST"])
def add_comment(doc_id: int):
    """
    Add a discussion comment or erratum note to a document.
    Request body: {"user_id": str, "comment_text": str}
    """
    document = db.get_or_404(Document, doc_id)
    data = request.get_json() or {}

    user_id = data.get("user_id", "anonymous").strip()
    comment_text = data.get("comment_text", "").strip()

    if not comment_text:
        return jsonify({
            "error": "Bad Request",
            "message": "Field 'comment_text' cannot be empty.",
        }), 400

    comment = Comment(
        document_id=document.id,
        user_id=user_id,
        comment_text=comment_text,
    )
    db.session.add(comment)
    db.session.commit()

    return jsonify({
        "message": "Comment added successfully.",
        "comment": comment.to_dict(),
    }), 201


@documents_bp.route("/<int:doc_id>/comments", methods=["GET"])
def get_comments(doc_id: int):
    """List all comments for a document."""
    document = db.get_or_404(Document, doc_id)
    comments = (
        Comment.query.filter_by(document_id=document.id)
        .order_by(Comment.created_at.asc())
        .all()
    )
    return jsonify({
        "document_id": doc_id,
        "count": len(comments),
        "comments": [c.to_dict() for c in comments],
    }), 200
