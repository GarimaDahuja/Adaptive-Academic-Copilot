"""
Entry point for running the Flask development server.
Usage:
    python backend/run.py
"""

from backend.app import create_app
from backend.app.config import Config

app = create_app(Config)

if __name__ == "__main__":
    port = app.config.get("PORT", 5000)
    debug = app.config.get("DEBUG", True)
    print(f"Starting Adaptive Academic Copilot API server on http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=debug)

