import React, { useState } from 'react';
import { api } from '../api';
import { Sparkles, Send, BookOpen, CheckCircle, ChevronDown, ChevronUp } from 'lucide-react';

export default function AskAI() {
  const [question, setQuestion] = useState('');
  const [subject, setSubject] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [showSources, setShowSources] = useState(false);

  const sampleQuestions = [
    { q: "What is Round Robin scheduling?", s: "Operating Systems" },
    { q: "What is a semaphore and what are wait() and signal() operations?", s: "Operating Systems" },
    { q: "What are the four necessary conditions for Deadlock?", s: "Operating Systems" },
    { q: "Explain the difference between 2NF and 3NF.", s: "Databases" }
  ];

  const handleAsk = async (e) => {
    if (e) e.preventDefault();
    if (!question.trim()) return;

    try {
      setLoading(true);
      setError(null);
      const res = await api.askQuestion({
        question: question.trim(),
        subject: subject.trim() || undefined,
        topK: 4,
      });
      setResponse(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleClick = (sample) => {
    setQuestion(sample.q);
    setSubject(sample.s);
  };

  return (
    <div className="chat-container">
      {/* Intro Header */}
      <div className="card">
        <h2 className="card-title">
          <Sparkles size={20} color="var(--primary)" /> Contextual Q&A with Gemini & RAG
        </h2>
        <p className="card-desc">
          Ask questions against your uploaded academic materials. Every answer is strictly grounded with page-level citations.
        </p>

        {/* Quick Sample Prompts */}
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', alignSelf: 'center' }}>Try asking:</span>
          {sampleQuestions.map((sample, idx) => (
            <button
              key={idx}
              className="btn btn-outline"
              style={{ padding: '0.3rem 0.75rem', fontSize: '0.8rem', borderRadius: '999px' }}
              onClick={() => handleSampleClick(sample)}
            >
              {sample.q}
            </button>
          ))}
        </div>

        {/* Question Form */}
        <form onSubmit={handleAsk}>
          <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '0.75rem' }}>
            <input
              type="text"
              className="form-input"
              style={{ width: '220px' }}
              placeholder="Subject (e.g. Operating Systems)"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
            />
          </div>

          <div className="chat-input-row">
            <input
              type="text"
              className="form-input"
              placeholder="Ask an academic question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              required
            />
            <button type="submit" className="btn btn-primary" disabled={loading} style={{ minWidth: '120px' }}>
              <Send size={16} /> {loading ? 'Searching...' : 'Ask AI'}
            </button>
          </div>
        </form>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="card" style={{ borderLeft: '4px solid var(--danger)', color: 'var(--danger)' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Answer Display */}
      {response && (
        <div className="answer-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Sparkles size={18} color="var(--primary)" />
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Grounded Academic Answer</h3>
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Confidence: {Math.round(response.confidence_score * 100)}% | Model: {response.model_used}
            </span>
          </div>

          {/* Answer Text */}
          <div style={{ fontSize: '0.95rem', lineHeight: '1.6', whiteSpace: 'pre-wrap', marginBottom: '1.25rem' }}>
            {response.answer}
          </div>

          {/* Citations */}
          {response.citations && response.citations.length > 0 && (
            <div className="citations-section">
              <h4 style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>
                Source Document Citations
              </h4>
              <div>
                {response.citations.map((c, idx) => (
                  <span key={idx} className="citation-chip">
                    <BookOpen size={12} />
                    {c.document_title} (Page {c.page_number || 'N/A'})
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Collapsible Source Chunks */}
          {response.sources_used && response.sources_used.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <button
                className="btn btn-outline"
                style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }}
                onClick={() => setShowSources(!showSources)}
              >
                {showSources ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                {showSources ? 'Hide Source Context' : 'View Extracted Source Passages'}
              </button>

              {showSources && (
                <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {response.sources_used.map((source, idx) => (
                    <div key={idx} style={{ background: 'var(--bg)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem' }}>
                      <div style={{ fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                        Source Chunk {idx + 1} (Page {source.page_number || 'N/A'})
                      </div>
                      <p style={{ color: 'var(--text-main)' }}>{source.content}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

