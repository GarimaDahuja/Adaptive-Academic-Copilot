import React from 'react';
import { BookOpen, Sparkles, Brain, CheckSquare } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, studentId, setStudentId }) {
  const tabs = [
    { id: 'documents', label: 'PDF Library', icon: BookOpen },
    { id: 'ask-ai', label: 'Ask AI (Gemini)', icon: Sparkles },
    { id: 'mastery', label: 'Adaptive Study Plan', icon: Brain },
    { id: 'quiz', label: 'Practice Quizzes', icon: CheckSquare },
  ];

  return (
    <header className="header">
      <div className="brand">
        <div className="brand-icon">
          <BookOpen size={22} />
        </div>
        <div>
          <span className="brand-title">Adaptive Academic Copilot</span>
          <span className="brand-badge">RL + RAG</span>
        </div>
      </div>

      <nav className="nav-tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              className={`nav-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      <div className="header-meta">
        <div className="student-tag">
          Student ID: <span>{studentId}</span>
        </div>
      </div>
    </header>
  );
}

