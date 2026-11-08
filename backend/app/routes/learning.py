"""
Learning and Mastery REST API endpoints.
Provides quiz access, submission grading, student mastery analytics,
spaced repetition schedules, and RL adaptive recommendations.
"""

from typing import List
from flask import Blueprint, jsonify, request

from backend.app.database import db
from backend.app.models.quiz import QuizQuestion, StudentMastery, DEFAULT_TOPIC_NAMES
from backend.app.services.mastery_service import (
    get_or_create_mastery_profile,
    get_mastery_vector,
    record_quiz_submission,
    get_student_analytics,
)
from rl.service import AdaptiveRecommendationService
from rl.policies.q_learning import QLearningPolicy

learning_bp = Blueprint("learning", __name__, url_prefix="/api/learning")


@learning_bp.route("/quizzes", methods=["POST"])
def create_quiz():
    """
    Create a new academic multiple-choice quiz question.
    """
    data = request.get_json() or {}

    subject = data.get("subject", "").strip()
    topic_index = data.get("topic_index", None)
    topic_name = data.get("topic_name", "").strip()
    question_text = data.get("question_text", "").strip()
    options = data.get("options", [])
    correct_option = data.get("correct_option", None)
    explanation = data.get("explanation", "").strip()
    difficulty = data.get("difficulty", "medium").strip().lower()

    if not subject:
        return jsonify({"error": "Bad Request", "message": "Field 'subject' is required."}), 400
    if not question_text:
        return jsonify({"error": "Bad Request", "message": "Field 'question_text' is required."}), 400
    if not isinstance(options, list) or len(options) < 2:
        return jsonify({
            "error": "Bad Request",
            "message": "Field 'options' must be a list with at least 2 choices.",
        }), 400
    if correct_option is None or not (0 <= int(correct_option) < len(options)):
        return jsonify({
            "error": "Bad Request",
            "message": f"Field 'correct_option' must be an integer index between 0 and {len(options) - 1}.",
        }), 400

    topic_idx = 0
    if topic_index is not None:
        try:
            topic_idx = int(topic_index)
            if topic_idx < 0 or topic_idx > 4:
                topic_idx = 0
        except ValueError:
            topic_idx = 0

    if not topic_name:
        topic_name = DEFAULT_TOPIC_NAMES[topic_idx]

    quiz = QuizQuestion(
        subject=subject,
        topic_index=topic_idx,
        topic_name=topic_name,
        question_text=question_text,
        options=options,
        correct_option=int(correct_option),
        explanation=explanation,
        difficulty=difficulty,
    )
    db.session.add(quiz)
    db.session.commit()

    return jsonify({
        "message": "Quiz question created successfully.",
        "quiz": quiz.to_dict(include_answer=True),
    }), 201


@learning_bp.route("/quizzes", methods=["GET"])
def list_quizzes():
    """
    List quiz questions with optional filtering.
    Filters: ?subject=...&topic_index=...&difficulty=...
    """
    query = QuizQuestion.query

    subject = request.args.get("subject")
    if subject:
        query = query.filter(QuizQuestion.subject.ilike(f"%{subject.strip()}%"))

    topic_index = request.args.get("topic_index")
    if topic_index is not None:
        try:
            query = query.filter(QuizQuestion.topic_index == int(topic_index))
        except ValueError:
            pass

    difficulty = request.args.get("difficulty")
    if difficulty:
        query = query.filter(QuizQuestion.difficulty == difficulty.strip().lower())

    quizzes = query.order_by(QuizQuestion.topic_index.asc(), QuizQuestion.id.asc()).all()
    include_answer = request.args.get("include_answer", "false").lower() in ("true", "1")

    return jsonify({
        "count": len(quizzes),
        "quizzes": [q.to_dict(include_answer=include_answer) for q in quizzes],
    }), 200


@learning_bp.route("/quizzes/<int:quiz_id>", methods=["GET"])
def get_quiz(quiz_id: int):
    """Retrieve a single quiz question without exposing answer key."""
    quiz = db.get_or_404(QuizQuestion, quiz_id)
    include_answer = request.args.get("include_answer", "false").lower() in ("true", "1")
    return jsonify({"quiz": quiz.to_dict(include_answer=include_answer)}), 200


@learning_bp.route("/quizzes/<int:quiz_id>/submit", methods=["POST"])
def submit_quiz_answer(quiz_id: int):
    """
    Submit student answer to a quiz.
    Evaluates correctness, logs attempt, and updates student mastery & spaced repetition schedule.
    """
    quiz = db.get_or_404(QuizQuestion, quiz_id)
    data = request.get_json() or {}

    student_id = data.get("student_id", "").strip()
    selected_option = data.get("selected_option", None)

    if not student_id:
        return jsonify({"error": "Bad Request", "message": "Field 'student_id' is required."}), 400
    if selected_option is None or not (0 <= int(selected_option) < len(quiz.options)):
        return jsonify({
            "error": "Bad Request",
            "message": f"Field 'selected_option' must be an integer between 0 and {len(quiz.options) - 1}.",
        }), 400

    result = record_quiz_submission(
        student_id=student_id,
        quiz_question=quiz,
        selected_option=int(selected_option),
    )

    return jsonify({
        "message": "Answer evaluated successfully.",
        "result": result,
    }), 200


@learning_bp.route("/mastery/<student_id>", methods=["GET"])
def get_mastery(student_id: str):
    """
    Retrieve a student's comprehensive skill mastery profile, accuracy, and spaced repetition alerts.
    """
    subject = request.args.get("subject", "General")
    analytics = get_student_analytics(student_id=student_id, subject=subject)
    return jsonify(analytics), 200


@learning_bp.route("/recommendations/<student_id>", methods=["GET"])
def get_adaptive_recommendation(student_id: str):
    """
    Call the Reinforcement Learning agent using the student's live mastery profile
    to generate an adaptive next-topic recommendation.
    """
    subject = request.args.get("subject", "General")
    mastery_vector = get_mastery_vector(student_id=student_id, subject=subject)

    # Initialize RL agent (Q-Learning policy)
    policy = QLearningPolicy(n_actions=len(DEFAULT_TOPIC_NAMES), seed=42)
    service = AdaptiveRecommendationService(
        policy=policy,
        topic_names=DEFAULT_TOPIC_NAMES,
    )

    rec = service.recommend(mastery_vector)

    # Fetch available quizzes for the recommended topic
    recommended_topic_idx = rec["recommended_topic_index"]
    suggested_quizzes = (
        QuizQuestion.query.filter_by(topic_index=recommended_topic_idx)
        .limit(3)
        .all()
    )

    rec["suggested_quizzes"] = [q.to_dict(include_answer=False) for q in suggested_quizzes]
    rec["student_id"] = student_id
    rec["subject"] = subject
    rec["current_full_mastery_profile"] = mastery_vector

    return jsonify(rec), 200

