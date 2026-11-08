from flask import Flask, jsonify
from backend.app.config import Config


def create_app(config_class=Config):
    """
    Application factory pattern.
    Creates, configures, and returns an instance of the Flask application.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Register blueprints
    from backend.app.routes.health import health_bp
    app.register_blueprint(health_bp)

    # Error handlers returning structured JSON responses
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "error": "Not Found",
            "message": "The requested resource could not be found."
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected server error occurred."
        }), 500

    return app
