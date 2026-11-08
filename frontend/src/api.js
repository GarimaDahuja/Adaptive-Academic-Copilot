/**
 * Unified API Client for Adaptive Academic Copilot Flask backend.
 */

const API_BASE = '/api';

async function handleResponse(response) {
  if (!response.ok) {
    let errorMsg = `HTTP Error ${response.status}`;
    try {
      const errorData = await response.json();
      errorMsg = errorData.message || errorData.error || errorMsg;
    } catch {
      // Fallback if response is not JSON
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

export const api = {
  // System Health
  getHealth: () => fetch(`${API_BASE}/health`).then(handleResponse),

  // Document Management
  getDocuments: (filters = {}) => {
    const params = new URLSearchParams();
    if (filters.subject) params.append('subject', filters.subject);
    if (filters.file_type) params.append('file_type', filters.file_type);
    if (filters.semester) params.append('semester', filters.semester);
    if (filters.verification_status) params.append('verification_status', filters.verification_status);
    const query = params.toString() ? `?${params.toString()}` : '';
    return fetch(`${API_BASE}/documents${query}`).then(handleResponse);
  },

  uploadDocument: (formData) => {
    return fetch(`${API_BASE}/documents/upload`, {
      method: 'POST',
      body: formData,
    }).then(handleResponse);
  },

  voteDocument: (docId, voteType) => {
    return fetch(`${API_BASE}/documents/${docId}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vote_type: voteType }),
    }).then(handleResponse);
  },

  verifyDocument: (docId, status) => {
    return fetch(`${API_BASE}/documents/${docId}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    }).then(handleResponse);
  },

  getComments: (docId) => {
    return fetch(`${API_BASE}/documents/${docId}/comments`).then(handleResponse);
  },

  postComment: (docId, userId, commentText) => {
    return fetch(`${API_BASE}/documents/${docId}/comments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, comment_text: commentText }),
    }).then(handleResponse);
  },

  // Gemini RAG & Contextual Q&A
  askQuestion: ({ question, subject, documentIds, topK = 4 }) => {
    return fetch(`${API_BASE}/qa/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        subject,
        document_ids: documentIds,
        top_k: topK,
      }),
    }).then(handleResponse);
  },

  // Student Mastery, Quizzes, & RL Recommendations
  getMastery: (studentId, subject = 'General') => {
    return fetch(`${API_BASE}/learning/mastery/${encodeURIComponent(studentId)}?subject=${encodeURIComponent(subject)}`).then(handleResponse);
  },

  getRecommendations: (studentId, subject = 'General') => {
    return fetch(`${API_BASE}/learning/recommendations/${encodeURIComponent(studentId)}?subject=${encodeURIComponent(subject)}`).then(handleResponse);
  },

  getQuizzes: (filters = {}) => {
    const params = new URLSearchParams();
    if (filters.subject) params.append('subject', filters.subject);
    if (filters.topic_index !== undefined && filters.topic_index !== '') {
      params.append('topic_index', filters.topic_index);
    }
    if (filters.difficulty) params.append('difficulty', filters.difficulty);
    const query = params.toString() ? `?${params.toString()}` : '';
    return fetch(`${API_BASE}/learning/quizzes${query}`).then(handleResponse);
  },

  submitQuiz: (quizId, studentId, selectedOption) => {
    return fetch(`${API_BASE}/learning/quizzes/${quizId}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        student_id: studentId,
        selected_option: selectedOption,
      }),
    }).then(handleResponse);
  },
};

