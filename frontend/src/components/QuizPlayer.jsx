import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { CheckSquare, CheckCircle, XCircle, ArrowRight, RefreshCw, PlusCircle } from 'lucide-react';

export default function QuizPlayer({ studentId, initialTopicIndex, onMasteryUpdated }) {
  const [quizzes, setQuizzes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTopic, setSelectedTopic] = useState(initialTopicIndex !== undefined ? String(initialTopicIndex) : '');
  const [selectedDifficulty, setSelectedDifficulty] = useState('');

  // Active quiz state
  const [activeQuizIndex, setActiveQuizIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);

  const loadQuizzes = async () => {
    try {
      setLoading(true);
      const res = await api.getQuizzes({
        topic_index: selectedTopic !== '' ? selectedTopic : undefined,
        difficulty: selectedDifficulty !== '' ? selectedDifficulty : undefined,
      });
      setQuizzes(res.quizzes || []);
      setActiveQuizIndex(0);
      setSelectedOption(null);
      setResult(null);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadQuizzes();
  }, [selectedTopic, selectedDifficulty]);

  useEffect(() => {
    if (initialTopicIndex !== undefined) {
      setSelectedTopic(String(initialTopicIndex));
    }
  }, [initialTopicIndex]);

  const handleSubmit = async () => {
    if (selectedOption === null || !currentQuiz) return;

    try {
      setSubmitting(true);
      const res = await api.submitQuiz(currentQuiz.id, studentId, selectedOption);
      setResult(res.result);
      if (onMasteryUpdated) {
        onMasteryUpdated();
      }
    } catch (err) {
      alert(`Submission failed: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const handleNext = () => {
    setSelectedOption(null);
    setResult(null);
    if (activeQuizIndex < quizzes.length - 1) {
      setActiveQuizIndex(activeQuizIndex + 1);
    } else {
      setActiveQuizIndex(0);
    }
  };

  const seedSampleQuizzes = async () => {
    const sampleQuestions = [
      {
        subject: "Operating Systems",
        topic_index: 0,
        topic_name: "Foundations & Prerequisite Concepts",
        question_text: "What is the primary function of an operating system kernel?",
        options: [
          "Compiling high-level source code to binary",
          "Managing hardware resources and process execution",
          "Rendering user interface graphics",
          "Executing SQL database queries"
        ],
        correct_option: 1,
        explanation: "The kernel is the privileged core of the OS responsible for CPU scheduling, memory management, and hardware abstraction.",
        difficulty: "easy"
      },
      {
        subject: "Operating Systems",
        topic_index: 1,
        topic_name: "Core Data Structures & Principles",
        question_text: "Which scheduling algorithm is designed specifically for time-sharing systems?",
        options: [
          "First-Come, First-Served (FCFS)",
          "Shortest Job First (SJF)",
          "Round Robin (RR)",
          "Priority Scheduling"
        ],
        correct_option: 2,
        explanation: "Round Robin uses preemption and a fixed time slice (quantum) to ensure all active processes share the CPU fairly.",
        difficulty: "easy"
      },
      {
        subject: "Operating Systems",
        topic_index: 2,
        topic_name: "Intermediate Algorithms & Problem Solving",
        question_text: "What are the four necessary and sufficient conditions for deadlock to occur?",
        options: [
          "Mutual exclusion, Hold and wait, No preemption, Circular wait",
          "Mutual exclusion, Preemption, Paging, Fragmentation",
          "Starvation, Thrashing, Aging, Semaphore",
          "Critical section, Race condition, Mutex, Monitor"
        ],
        correct_option: 0,
        explanation: "Coffman conditions: Deadlock occurs if and only if mutual exclusion, hold and wait, no preemption, and circular wait hold simultaneously.",
        difficulty: "medium"
      },
      {
        subject: "Operating Systems",
        topic_index: 3,
        topic_name: "Advanced System Architecture",
        question_text: "What is the primary cause of 'Thrashing' in a virtual memory system?",
        options: [
          "CPU clock speed being too high",
          "The system spending more time servicing page faults than executing instructions",
          "Hard drive fragmentation reaching critical levels",
          "Network packet loss over the loopback interface"
        ],
        correct_option: 1,
        explanation: "Thrashing occurs when active working sets exceed physical memory, causing continuous page swapping and near-zero CPU throughput.",
        difficulty: "hard"
      }
    ];

    try {
      setLoading(true);
      for (const q of sampleQuestions) {
        await api.getQuizzes({ subject: q.subject }); // check
        await fetch('/api/learning/quizzes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(q)
        });
      }
      alert('Sample quiz questions added successfully!');
      loadQuizzes();
    } catch (err) {
      alert(`Failed to add sample questions: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const currentQuiz = quizzes[activeQuizIndex];

  return (
    <div>
      {/* Filter and Topic Selection Bar */}
      <div className="card" style={{ padding: '1rem 1.5rem', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Filter Quizzes:</span>
            <select
              className="form-select"
              style={{ width: '220px' }}
              value={selectedTopic}
              onChange={(e) => setSelectedTopic(e.target.value)}
            >
              <option value="">All Syllabus Topics</option>
              <option value="0">Topic 1: Foundations</option>
              <option value="1">Topic 2: Core Data Structures</option>
              <option value="2">Topic 3: Intermediate Problem Solving</option>
              <option value="3">Topic 4: Advanced Architecture</option>
              <option value="4">Topic 5: Comprehensive Mastery</option>
            </select>

            <select
              className="form-select"
              style={{ width: '150px' }}
              value={selectedDifficulty}
              onChange={(e) => setSelectedDifficulty(e.target.value)}
            >
              <option value="">All Difficulties</option>
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>

          <button className="btn btn-outline" style={{ fontSize: '0.8rem' }} onClick={seedSampleQuizzes}>
            <PlusCircle size={14} /> Seed Sample Questions
          </button>
        </div>
      </div>

      {/* Quiz Card */}
      {loading ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading quizzes...</p>
      ) : quizzes.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
          <CheckSquare size={48} color="var(--text-muted)" style={{ margin: '0 auto 1rem' }} />
          <h3>No quizzes available</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem', marginBottom: '1.5rem' }}>
            Click below to populate the database with academic sample questions.
          </p>
          <button className="btn btn-primary" onClick={seedSampleQuizzes}>
            <PlusCircle size={16} /> Add Sample Questions
          </button>
        </div>
      ) : currentQuiz ? (
        <div className="card">
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <span className="brand-badge" style={{ textTransform: 'capitalize' }}>
                {currentQuiz.difficulty} Difficulty
              </span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
                {currentQuiz.topic_name}
              </span>
            </div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Question {activeQuizIndex + 1} of {quizzes.length}
            </span>
          </div>

          {/* Question Text */}
          <h3 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '1.5rem', lineHeight: '1.5' }}>
            {currentQuiz.question_text}
          </h3>

          {/* Options */}
          <div style={{ marginBottom: '1.5rem' }}>
            {currentQuiz.options.map((option, idx) => {
              let optionClass = 'quiz-option';
              if (result !== null) {
                if (idx === result.correct_option) {
                  optionClass += ' correct';
                } else if (idx === selectedOption && !result.is_correct) {
                  optionClass += ' incorrect';
                }
              } else if (idx === selectedOption) {
                optionClass += ' active';
              }

              const letters = ['A', 'B', 'C', 'D'];

              return (
                <button
                  key={idx}
                  className={optionClass}
                  onClick={() => !result && setSelectedOption(idx)}
                  disabled={result !== null}
                >
                  <div className="option-badge">{letters[idx]}</div>
                  <span>{option}</span>
                </button>
              );
            })}
          </div>

          {/* Action / Next Button */}
          {!result ? (
            <button
              className="btn btn-primary"
              disabled={selectedOption === null || submitting}
              onClick={handleSubmit}
            >
              {submitting ? 'Evaluating...' : 'Submit Answer'}
            </button>
          ) : (
            <div>
              {/* Result Feedback Banner */}
              <div
                style={{
                  background: result.is_correct ? 'var(--success-light)' : 'var(--danger-light)',
                  border: `1px solid ${result.is_correct ? 'var(--success)' : 'var(--danger)'}`,
                  borderRadius: 'var(--radius)',
                  padding: '1rem',
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem',
                }}
              >
                {result.is_correct ? (
                  <CheckCircle size={22} color="var(--success)" style={{ flexShrink: 0 }} />
                ) : (
                  <XCircle size={22} color="var(--danger)" style={{ flexShrink: 0 }} />
                )}
                <div>
                  <h4 style={{ fontWeight: 600, color: result.is_correct ? '#065f46' : '#991b1b' }}>
                    {result.is_correct ? 'Correct! Outstanding work.' : 'Incorrect.'}
                  </h4>
                  <p style={{ fontSize: '0.9rem', marginTop: '0.25rem', color: 'var(--text-main)' }}>
                    {result.explanation}
                  </p>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                    Updated Topic Mastery: <strong>{Math.round(result.updated_mastery * 100)}%</strong> | Next scheduled review: {new Date(result.next_revision_at).toLocaleDateString()}
                  </div>
                </div>
              </div>

              <button className="btn btn-primary" onClick={handleNext}>
                Next Question <ArrowRight size={16} />
              </button>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

