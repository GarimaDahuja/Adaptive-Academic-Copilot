import React, { useState } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import DocumentManager from './components/DocumentManager';
import AskAI from './components/AskAI';
import MasteryDashboard from './components/MasteryDashboard';
import QuizPlayer from './components/QuizPlayer';

export default function App() {
  const [activeTab, setActiveTab] = useState('mastery');
  const [studentId] = useState('student_001');
  const [masteryKey, setMasteryKey] = useState(0);
  const [recommendedTopicIndex, setRecommendedTopicIndex] = useState(undefined);

  const handleMasteryUpdated = () => {
    setMasteryKey((k) => k + 1);
  };

  const handleSelectTopicForQuiz = (topicIndex) => {
    setRecommendedTopicIndex(topicIndex);
    setActiveTab('quiz');
  };

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        studentId={studentId}
      />

      <main className="main-content">
        {activeTab === 'documents' && (
          <DocumentManager studentId={studentId} />
        )}

        {activeTab === 'ask-ai' && (
          <AskAI />
        )}

        {activeTab === 'mastery' && (
          <MasteryDashboard
            key={masteryKey}
            studentId={studentId}
            onSelectTopicForQuiz={handleSelectTopicForQuiz}
          />
        )}

        {activeTab === 'quiz' && (
          <QuizPlayer
            studentId={studentId}
            initialTopicIndex={recommendedTopicIndex}
            onMasteryUpdated={handleMasteryUpdated}
          />
        )}
      </main>
    </div>
  );
}
