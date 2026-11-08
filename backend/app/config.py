import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base application configuration loaded from environment variables."""
    SECRET_KEY = os.getenv("SECRET_KEY", "default-dev-secret-key-change-me")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "t")
    PORT = int(os.getenv("PORT", 5000))

    # Database configuration (PostgreSQL + pgvector for upcoming milestone)
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/academic_copilot"
    )

    # Gemini API configuration for upcoming milestone
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class TestConfig(Config):
    """Configuration used during automated tests."""
    TESTING = True
    DEBUG = False

