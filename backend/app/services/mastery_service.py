"""
Student Mastery and Spaced Repetition business logic.
Maintains student skill estimates, calculates revision intervals,
and bridges student profiles with RL adaptive recommendations.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from backend.app.database import db
from backend.app.models.quiz import (
    QuizQuestion,
    StudentMastery,
    QuizAttempt,
    DEFAULT_TOPIC_NAMES,
)


def get_or_create_mastery_profile(
    student_id: str,
    subject: str = "General",
) -> List[StudentMastery]:
    """
    Ensure the student has a mastery record for all 5 standard syllabus topics.
    Initializes missing topics with a baseline score of 0.15.
    """
    existing_records = (
        StudentMastery.query.filter_by(student_id=student_id, subject=subject)
        .order_by(StudentMastery.topic_index.asc())
        .all()
    )
    existing_by_topic = {r.topic_index: r for r in existing_records}

    records_to_add = []
    for idx, name in enumerate(DEFAULT_TOPIC_NAMES):
        if idx not in existing_by_topic:
            new_record = StudentMastery(
                student_id=student_id,
                subject=subject,
                topic_index=idx,
                topic_name=name,
                mastery_score=0.15,
                total_attempts=0,
                correct_attempts=0,
            )
            records_to_add.append(new_record)
            existing_by_topic[idx] = new_record

    if records_to_add:
        db.session.add_all(records_to_add)
        db.session.commit()

    return [existing_by_topic[i] for i in range(len(DEFAULT_TOPIC_NAMES))]


def get_mastery_vector(student_id: str, subject: str = "General") -> List[float]:
    """Return a 5-element float vector of the student's current topic masteries."""
    profile = get_or_create_mastery_profile(student_id, subject)
    return [round(r.mastery_score, 4) for r in profile]


def record_quiz_submission(
    student_id: str,
    quiz_question: QuizQuestion,
    selected_option: int,
) -> Dict[str, Any]:
    """
    Grade a submitted quiz answer, record the attempt, and update student mastery
    and spaced repetition intervals.
    """
    is_correct = bool(selected_option == quiz_question.correct_option)

    # 1. Log attempt
    attempt = QuizAttempt(
        student_id=student_id,
        quiz_id=quiz_question.id,
        selected_option=selected_option,
        is_correct=is_correct,
    )
    db.session.add(attempt)

    # 2. Update StudentMastery
    profile = get_or_create_mastery_profile(student_id, quiz_question.subject)
    target_topic_record = next(
        (r for r in profile if r.topic_index == quiz_question.topic_index),
        None,
    )

    if target_topic_record:
        target_topic_record.update_score(is_correct)

    db.session.commit()

    return {
        "is_correct": is_correct,
        "selected_option": selected_option,
        "correct_option": quiz_question.correct_option,
        "explanation": quiz_question.explanation,
        "topic_index": quiz_question.topic_index,
        "topic_name": quiz_question.topic_name,
        "updated_mastery": (
            round(target_topic_record.mastery_score, 3)
            if target_topic_record
            else None
        ),
        "next_revision_at": (
            target_topic_record.next_revision_at.isoformat()
            if target_topic_record and target_topic_record.next_revision_at
            else None
        ),
    }


def get_student_analytics(
    student_id: str,
    subject: str = "General",
) -> Dict[str, Any]:
    """
    Compile comprehensive analytics: overall mastery percentage, total quizzes,
    overall accuracy, and topics due for revision.
    """
    profile = get_or_create_mastery_profile(student_id, subject)

    total_attempts = sum(r.total_attempts for r in profile)
    correct_attempts = sum(r.correct_attempts for r in profile)
    overall_accuracy = (
        round((correct_attempts / total_attempts) * 100, 1)
        if total_attempts > 0
        else 0.0
    )
    avg_mastery = round(float(sum(r.mastery_score for r in profile) / len(profile)), 3)

    topics_due = [r.to_dict() for r in profile if r.to_dict()["is_due_for_revision"]]

    return {
        "student_id": student_id,
        "subject": subject,
        "average_mastery_score": avg_mastery,
        "average_mastery_percentage": round(avg_mastery * 100, 1),
        "total_quizzes_taken": total_attempts,
        "overall_accuracy_percentage": overall_accuracy,
        "topics_due_for_revision": topics_due,
        "topic_masteries": [r.to_dict() for r in profile],
    }
