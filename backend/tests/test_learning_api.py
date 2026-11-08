"""
Automated unit and integration tests for Quizzes, Student Mastery,
Spaced Repetition, and RL Adaptive Recommendations.
"""

import pytest
from backend.app.models.quiz import QuizQuestion, StudentMastery, QuizAttempt


class TestQuizManagementAPI:
    """Tests for creating and listing quizzes."""

    def test_create_quiz_valid(self, client):
        data = {
            "subject": "Operating Systems",
            "topic_index": 0,
            "topic_name": "Foundations & Prerequisite Concepts",
            "question_text": "What is the primary function of an operating system kernel?",
            "options": [
                "Compiling high-level source code",
                "Managing hardware resources and process execution",
                "Rendering user interface graphics",
                "Handling external database queries",
            ],
            "correct_option": 1,
            "explanation": "The kernel is the core component managing CPU, memory, and devices.",
            "difficulty": "easy",
        }
        res = client.post("/api/learning/quizzes", json=data)
        assert res.status_code == 201
        res_data = res.get_json()
        assert "quiz" in res_data
        assert res_data["quiz"]["question_text"] == data["question_text"]
        assert res_data["quiz"]["correct_option"] == 1

    def test_create_quiz_invalid_options(self, client):
        data = {
            "subject": "OS",
            "question_text": "Invalid question",
            "options": ["Single option"],
            "correct_option": 0,
        }
        res = client.post("/api/learning/quizzes", json=data)
        assert res.status_code == 400
        assert "at least 2 choices" in res.get_json()["message"]

    def test_list_quizzes_hides_answers_by_default(self, client, db_session):
        q = QuizQuestion(
            subject="Algorithms",
            topic_index=1,
            topic_name="Core Data Structures",
            question_text="What is the average time complexity of QuickSort?",
            options=["O(N)", "O(N log N)", "O(N^2)", "O(1)"],
            correct_option=1,
            explanation="QuickSort partitions in linear time, yielding O(N log N) average depth.",
            difficulty="medium",
        )
        db_session.add(q)
        db_session.commit()

        res = client.get("/api/learning/quizzes?subject=Algorithms")
        assert res.status_code == 200
        quizzes = res.get_json()["quizzes"]
        assert len(quizzes) >= 1

        target = next((item for item in quizzes if item["id"] == q.id), None)
        assert target is not None
        assert "correct_option" not in target  # Hidden from student

        # But visible if explicitly requested (e.g. for instructors)
        res_admin = client.get(f"/api/learning/quizzes?subject=Algorithms&include_answer=true")
        target_admin = next(
            (item for item in res_admin.get_json()["quizzes"] if item["id"] == q.id),
            None,
        )
        assert target_admin["correct_option"] == 1

    def test_get_single_quiz_by_id(self, client, db_session):
        q = QuizQuestion(
            subject="Computer Networks",
            topic_index=3,
            topic_name="Network Layer & Routing",
            question_text="Which protocol is used for finding IP to MAC address mapping?",
            options=["DNS", "ARP", "DHCP", "ICMP"],
            correct_option=1,
            explanation="ARP resolves IP addresses to physical MAC addresses.",
            difficulty="easy",
        )
        db_session.add(q)
        db_session.commit()

        # By default, hides correct option
        res = client.get(f"/api/learning/quizzes/{q.id}")
        assert res.status_code == 200
        quiz_data = res.get_json()["quiz"]
        assert quiz_data["id"] == q.id
        assert quiz_data["subject"] == "Computer Networks"
        assert "correct_option" not in quiz_data

        # With include_answer=true
        res_ans = client.get(f"/api/learning/quizzes/{q.id}?include_answer=true")
        assert res_ans.status_code == 200
        assert res_ans.get_json()["quiz"]["correct_option"] == 1

        # Non-existent quiz returns 404
        res_404 = client.get("/api/learning/quizzes/999999")
        assert res_404.status_code == 404


class TestMasteryAndGradingAPI:
    """Tests for quiz submission, mastery score calculation, and spaced repetition."""

    def test_submit_correct_answer_increases_mastery(self, client, db_session):
        q = QuizQuestion(
            subject="Databases",
            topic_index=2,
            topic_name="Intermediate Algorithms & Problem Solving",
            question_text="Which normal form eliminates transitive dependency?",
            options=["1NF", "2NF", "3NF", "BCNF"],
            correct_option=2,
            explanation="3NF eliminates transitive dependencies on candidate keys.",
        )
        db_session.add(q)
        db_session.commit()

        # Initial baseline mastery is 0.15
        student_id = "student_alpha"
        res = client.post(
            f"/api/learning/quizzes/{q.id}/submit",
            json={"student_id": student_id, "selected_option": 2},
        )
        assert res.status_code == 200
        data = res.get_json()["result"]

        assert data["is_correct"] is True
        assert data["selected_option"] == 2
        assert data["correct_option"] == 2
        # Mastery should have increased above 0.15
        assert data["updated_mastery"] > 0.15
        assert data["next_revision_at"] is not None

        # Verify database attempt
        attempts = QuizAttempt.query.filter_by(student_id=student_id).all()
        assert len(attempts) == 1
        assert attempts[0].is_correct is True

    def test_submit_incorrect_answer_penalizes_mastery(self, client, db_session):
        q = QuizQuestion(
            subject="Databases",
            topic_index=0,
            topic_name="Foundations",
            question_text="What does ACID stand for?",
            options=["Option A", "Option B", "Option C", "Option D"],
            correct_option=0,
        )
        db_session.add(q)
        db_session.commit()

        student_id = "student_beta"
        res = client.post(
            f"/api/learning/quizzes/{q.id}/submit",
            json={"student_id": student_id, "selected_option": 3},
        )
        assert res.status_code == 200
        data = res.get_json()["result"]

        assert data["is_correct"] is False
        # Baseline is 0.15; penalty should decrease score
        assert data["updated_mastery"] < 0.15


class TestAnalyticsAndRLRecommendationAPI:
    """Tests for student analytics and RL adaptive recommendation routes."""

    def test_get_mastery_analytics(self, client):
        student_id = "student_gamma"
        res = client.get(f"/api/learning/mastery/{student_id}?subject=OS")
        assert res.status_code == 200
        data = res.get_json()

        assert data["student_id"] == student_id
        assert data["subject"] == "OS"
        assert "average_mastery_percentage" in data
        assert "overall_accuracy_percentage" in data
        assert len(data["topic_masteries"]) == 5

    def test_get_adaptive_recommendation_from_rl(self, client):
        student_id = "student_delta"
        res = client.get(f"/api/learning/recommendations/{student_id}?subject=General")
        assert res.status_code == 200
        data = res.get_json()

        assert "recommended_topic_index" in data
        assert 0 <= data["recommended_topic_index"] < 5
        assert "recommended_topic_name" in data
        assert "rationale" in data
        assert "suggested_quizzes" in data
        assert "current_full_mastery_profile" in data
        assert len(data["current_full_mastery_profile"]) == 5

