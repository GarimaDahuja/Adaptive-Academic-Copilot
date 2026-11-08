import pytest
from backend.app import create_app
from backend.app.config import TestConfig
from backend.app.database import db


@pytest.fixture
def app():
    """Create and configure a clean Flask app instance with an in-memory SQLite database for each test."""
    app = create_app(TestConfig)

    with app.app_context():
        # Import models so SQLAlchemy registers them before create_all
        import backend.app.models  # noqa: F401
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """A test client for sending simulated HTTP requests to the app."""
    return app.test_client()


@pytest.fixture
def db_session(app):
    """Provides a transactional database session for tests."""
    with app.app_context():
        yield db.session
