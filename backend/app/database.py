"""
Database initialization module for Flask-SQLAlchemy.
Provides the central `db` instance used by all application models.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

