from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__, url_prefix="/api")


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    Used by frontend, uptime monitors, and tests to verify the API server is operational.
    """
    return jsonify({
        "status": "healthy",
        "service": "Adaptive Academic Copilot API",
        "version": "0.1.0"
    }), 200

