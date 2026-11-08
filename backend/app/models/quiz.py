"""
Quiz, Student Mastery, and Spaced Repetition database models.
Tracks questions, student accuracy, evolving knowledge mastery, and revision scheduling.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from backend.app.database import db

DEFAULT_TOPIC_NAMES = [
    "Foundations & Prerequisite Concepts",
    "Core Data Structures & Principles",
    "Intermediate Algorithms & Problem Solving",
    "Advanced System Architecture",
    "Comprehensive Applications & PYQ Mastery",
]


class QuizQuestion(db.Model):
    """
    Multiple-choice academic quiz question tied to a course topic and difficulty.
    """

    __tablename__ = "quiz_questions"

    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(100), nullable=False)
    topic_index = db.Column(db.Integer, nullable=False, default=0)
    topic_name = db.Column(db.String(200), nullable=False)
    question_text = db.Column(db.Text, nullable=False)

    # List of 4 option strings stored as JSON
    options = db.Column(db.JSON, nullable=False)
    correct_option = db.Column(db.Integer, nullable=False)  # 0, 1, 2, or 3
    explanation = db.Column(db.Text, nullable=True)
    difficulty = db.Column(db.String(20), nullable=False, default="medium")

    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    attempts = db.relationship(
        "QuizAttempt",
        backref="quiz_question",
        cascade="all, delete-orphan",
        lazy=True,
    )

    def to_dict(self, include_answer: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "subject": self.subject,
            "topic_index": self.topic_index,
            "topic_name": self.topic_name,
            "question_text": self.question_text,
            "options": self.options,
            "difficulty": self.difficulty,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_answer:
            data["correct_option"] = self.correct_option
            data["explanation"] = self.explanation
        return data


class StudentMastery(db.Model):
    """
    Tracks a student's evolving mastery score and spaced repetition schedule
    for a specific syllabus topic.
    """

    __tablename__ = "student_mastery"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(100), nullable=False, index=True)
    subject = db.Column(db.String(100), nullable=False, index=True)
    topic_index = db.Column(db.Integer, nullable=False)
    topic_name = db.Column(db.String(200), nullable=False)

    # Continuous mastery score in [0.0, 1.0]
    mastery_score = db.Column(db.Float, nullable=False, default=0.15)
    total_attempts = db.Column(db.Integer, nullable=False, default=0)
    correct_attempts = db.Column(db.Integer, nullable=False, default=0)

    # Spaced repetition tracking
    last_studied_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    next_revision_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    def update_score(self, is_correct: bool) -> float:
        """
        Update mastery score using an adaptive exponential update rule:
        - If correct: gains knowledge with diminishing returns as mastery approaches 1.0
        - If incorrect: penalizes score to signal need for review
        """
        self.total_attempts += 1
        now = datetime.now(timezone.utc)
        self.last_studied_at = now

        if is_correct:
            self.correct_attempts += 1
            gain = 0.25 * (1.0 - self.mastery_score)
            self.mastery_score = min(1.0, self.mastery_score + gain)
        else:
            penalty = 0.15 * self.mastery_score
            self.mastery_score = max(0.05, self.mastery_score - penalty)

        # Spaced repetition interval based on performance:
        # Mastery < 0.40 -> review in 1 day
        # Mastery 0.40 - 0.70 -> review in 3 days
        # Mastery >= 0.70 -> review in 7 days
        if self.mastery_score < 0.40:
            interval_days = 1
        elif self.mastery_score < 0.70:
            interval_days = 3
        else:
            interval_days = 7

        self.next_revision_at = now + timedelta(days=interval_days)
        return self.mastery_score

    def to_dict(self) -> Dict[str, Any]:
        accuracy = (
            round((self.correct_attempts / self.total_attempts) * 100, 1)
            if self.total_attempts > 0
            else 0.0
        )
        now = datetime.now(timezone.utc)
        # Handle SQLite timezone-naive datetimes gracefully
        revision_time = self.next_revision_at
        if revision_time and revision_time.tzinfo is None:
            revision_time = revision_time.replace(tzinfo=timezone.utc)

        is_due = bool(revision_time and revision_time <= now)

        return {
            "id": self.id,
            "student_id": self.student_id,
            "subject": self.subject,
            "topic_index": self.topic_index,
            "topic_name": self.topic_name,
            "mastery_score": round(self.mastery_score, 3),
            "mastery_percentage": round(self.mastery_score * 100, 1),
            "total_attempts": self.total_attempts,
            "correct_attempts": self.correct_attempts,
            "accuracy_percentage": accuracy,
            "last_studied_at": self.last_studied_at.isoformat() if self.last_studied_at else None,
            "next_revision_at": self.next_revision_at.isoformat() if self.next_revision_at else None,
            "is_due_for_revision": is_due,
        }


class QuizAttempt(db.Model):
    """
    Log of each individual quiz answer submitted by a student.
    """

    __tablename__ = "quiz_attempts"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(100), nullable=False, index=True)
    quiz_id = db.Column(
        db.Integer,
        db.ForeignKey("quiz_questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    selected_option = db.Column(db.Integer, nullable=False)
    is_correct = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "quiz_id": self.quiz_id,
            "selected_option": self.selected_option,
            "is_correct": self.is_correct,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
