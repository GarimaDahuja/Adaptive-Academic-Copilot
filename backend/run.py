"""
Entry point for running the Flask development server.
Usage (from project root):
    python backend/run.py
  or:
    python -m backend.run
"""

import sys
import os

# Ensure the project root (parent of this file's directory) is on sys.path
# so "from backend.app import ..." works regardless of how the script is invoked.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.app import create_app
from backend.app.config import Config
from backend.app.database import db

app = create_app(Config)

if __name__ == "__main__":
    # Ensure all tables exist (safe no-op if already created / PostgreSQL managed separately)
    with app.app_context():
        db.create_all()

    port = app.config.get("PORT", 5000)
    debug = app.config.get("DEBUG", True)
    db_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    db_label = "SQLite (local)" if db_uri.startswith("sqlite") else "PostgreSQL"
    print(f"🗄️  Database : {db_label}")
    print(f"🚀 Starting Adaptive Academic Copilot API on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=debug)

