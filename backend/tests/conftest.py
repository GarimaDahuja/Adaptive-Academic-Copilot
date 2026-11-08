import pytest
from backend.app import create_app
from backend.app.config import TestConfig


@pytest.fixture
def app():
    """Create and configure a clean Flask app instance for each test."""
    app = create_app(TestConfig)
    return app


@pytest.fixture
def client(app):
    """A test client for sending simulated HTTP requests to the app."""
    return app.test_client()
