import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Brain, AlertCircle, Award, Target, Compass, ArrowRight } from 'lucide-react';

export default function MasteryDashboard({ studentId, onSelectTopicForQuiz }) {
  const [analytics, setAnalytics] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [subject, setSubject] = useState('General');

  const loadData = async () => {
    try {
      setLoading(true);
      const [analyticsData, recData] = await Promise.all([
        api.getMastery(studentId, subject),
        api.getRecommendations(studentId, subject),
      ]);
      setAnalytics(analyticsData);
      setRecommendation(recData);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [studentId, subject]);

  if (loading) {
    return <p style={{ color: 'var(--text-muted)' }}>Loading mastery analytics & RL policy recommendations...</p>;
  }

  const topicsDue = analytics?.topics_due_for_revision || [];

  return (
    <div>
      {/* Spaced Repetition Due Alert */}
      {topicsDue.length > 0 && (
        <div className="revision-alert">
          <AlertCircle size={22} color="var(--warning)" style={{ flexShrink: 0 }} />
          <div>
            <strong>Spaced Repetition Alert:</strong> You have {topicsDue.length} topic(s) due for scheduled review today!
            <div style={{ fontSize: '0.85rem', marginTop: '0.2rem' }}>
              Due: {topicsDue.map((t) => t.topic_name).join(', ')}
            </div>
          </div>
        </div>
      )}

      {/* RL Agent Recommendation Card */}
      {recommendation && (
        <div className="rl-card">
          <div className="rl-badge">
            <Compass size={14} /> RL Agent Recommendation
          </div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.4rem', color: 'var(--text-main)' }}>
            Recommended Focus: {recommendation.recommended_topic_name}
          </h3>
          <p style={{ fontSize: '0.95rem', color: '#4338ca', marginBottom: '1rem', fontWeight: 500 }}>
            {recommendation.rationale}
          </p>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Current Mastery: <strong>{Math.round(recommendation.current_mastery * 100)}%</strong> | Policy: Q-Learning (TD)
            </span>

            {onSelectTopicForQuiz && (
              <button
                className="btn btn-primary"
                onClick={() => onSelectTopicForQuiz(recommendation.recommended_topic_index)}
              >
                Practice Recommended Topic <ArrowRight size={15} />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Analytics Summary Stats */}
      {analytics && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
          <div className="card" style={{ marginBottom: 0, textAlign: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Overall Mastery</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--primary)' }}>
              {analytics.average_mastery_percentage}%
            </div>
          </div>

          <div className="card" style={{ marginBottom: 0, textAlign: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Quizzes Attempted</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-main)' }}>
              {analytics.total_quizzes_taken}
            </div>
          </div>

          <div className="card" style={{ marginBottom: 0, textAlign: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Accuracy Rate</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--success)' }}>
              {analytics.overall_accuracy_percentage}%
            </div>
          </div>
        </div>
      )}

      {/* Topic Mastery Progress Bars */}
      <div className="card">
        <h3 className="card-title">
          <Brain size={20} color="var(--primary)" /> Syllabus Mastery & Spaced Repetition Trajectory
        </h3>
        <p className="card-desc">
          Calculated dynamically by the adaptive skill estimator after each quiz submission.
        </p>

        {analytics?.topic_masteries?.map((t) => {
          const percentage = Math.round(t.mastery_score * 100);
          return (
            <div key={t.topic_index} className="mastery-row">
              <div className="mastery-header">
                <span>
                  <strong>Topic {t.topic_index + 1}:</strong> {t.topic_name}
                </span>
                <span>
                  <strong>{percentage}%</strong> ({t.correct_attempts}/{t.total_attempts} correct)
                </span>
              </div>

              <div className="progress-bar-bg">
                <div
                  className="progress-bar-fill"
                  style={{
                    width: `${percentage}%`,
                    backgroundColor:
                      percentage >= 70
                        ? 'var(--success)'
                        : percentage >= 40
                        ? 'var(--primary)'
                        : 'var(--warning)',
                  }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                <span>Status: {percentage >= 70 ? 'Proficient' : percentage >= 40 ? 'Developing' : 'Novice'}</span>
                <span>
                  Next Scheduled Review:{' '}
                  {t.next_revision_at ? new Date(t.next_revision_at).toLocaleDateString() : 'N/A'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

