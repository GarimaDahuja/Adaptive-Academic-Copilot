import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Upload, ThumbsUp, ThumbsDown, MessageSquare, CheckCircle, Clock, AlertTriangle, FileText } from 'lucide-react';

export default function DocumentManager({ studentId }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [filterSubject, setFilterSubject] = useState('');
  const [filterType, setFilterType] = useState('');

  // Upload Form State
  const [title, setTitle] = useState('');
  const [subject, setSubject] = useState('');
  const [semester, setSemester] = useState('4');
  const [fileType, setFileType] = useState('notes');
  const [selectedFile, setSelectedFile] = useState(null);

  // Comments modal state
  const [activeDocForComments, setActiveDocForComments] = useState(null);
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState('');

  const loadDocuments = async () => {
    try {
      setLoading(true);
      const res = await api.getDocuments({
        subject: filterSubject,
        file_type: filterType,
      });
      setDocuments(res.documents || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, [filterSubject, filterType]);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      alert('Please choose a PDF file to upload.');
      return;
    }

    try {
      setUploading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('title', title);
      formData.append('subject', subject);
      formData.append('semester', semester);
      formData.append('file_type', fileType);
      formData.append('uploader_id', studentId);

      await api.uploadDocument(formData);
      alert('Document uploaded and chunks extracted successfully!');
      setTitle('');
      setSubject('');
      setSelectedFile(null);
      // Reset file input
      document.getElementById('pdf-file-input').value = '';
      loadDocuments();
    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleVote = async (docId, voteType) => {
    try {
      await api.voteDocument(docId, voteType);
      loadDocuments();
    } catch (err) {
      alert(`Vote failed: ${err.message}`);
    }
  };

  const handleVerify = async (docId, newStatus) => {
    try {
      await api.verifyDocument(docId, newStatus);
      loadDocuments();
    } catch (err) {
      alert(`Verification update failed: ${err.message}`);
    }
  };

  const openComments = async (doc) => {
    setActiveDocForComments(doc);
    try {
      const res = await api.getComments(doc.id);
      setComments(res.comments || []);
    } catch (err) {
      console.error(err);
    }
  };

  const handleAddComment = async (e) => {
    e.preventDefault();
    if (!newComment.trim() || !activeDocForComments) return;
    try {
      await api.postComment(activeDocForComments.id, studentId, newComment);
      setNewComment('');
      const res = await api.getComments(activeDocForComments.id);
      setComments(res.comments || []);
    } catch (err) {
      alert(`Failed to add comment: ${err.message}`);
    }
  };

  return (
    <div>
      {/* Upload Form Card */}
      <div className="card">
        <h2 className="card-title">
          <Upload size={20} color="var(--primary)" /> Upload Academic Material
        </h2>
        <p className="card-desc">
          Upload semester PYQs, lab manuals, or lecture notes. The backend automatically extracts text chunks for RAG.
        </p>

        <form onSubmit={handleUpload}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
            <div className="form-group">
              <label className="form-label">Title</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Operating Systems End-Sem 2024"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Subject</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Operating Systems"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Material Type</label>
              <select className="form-select" value={fileType} onChange={(e) => setFileType(e.target.value)}>
                <option value="pyq">Previous Year Questions (PYQ)</option>
                <option value="lab_manual">Lab Manual</option>
                <option value="notes">Lecture Notes</option>
                <option value="syllabus">Syllabus</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Semester</label>
              <select className="form-select" value={semester} onChange={(e) => setSemester(e.target.value)}>
                {[1, 2, 3, 4, 5, 6, 7, 8].map((s) => (
                  <option key={s} value={s}>Semester {s}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-group" style={{ marginTop: '0.5rem' }}>
            <label className="form-label">Select PDF Document</label>
            <input
              id="pdf-file-input"
              type="file"
              accept=".pdf"
              className="form-input"
              onChange={(e) => setSelectedFile(e.target.files[0])}
              required
            />
          </div>

          <button type="submit" className="btn btn-primary" disabled={uploading}>
            <Upload size={16} />
            {uploading ? 'Processing & Chunking PDF...' : 'Upload & Extract Chunks'}
          </button>
        </form>
      </div>

      {/* Filter and Library Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 600 }}>
          Academic Materials Library ({documents.length})
        </h3>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <input
            type="text"
            className="form-input"
            style={{ width: '180px' }}
            placeholder="Filter by subject..."
            value={filterSubject}
            onChange={(e) => setFilterSubject(e.target.value)}
          />

          <select className="form-select" style={{ width: '150px' }} value={filterType} onChange={(e) => setFilterType(e.target.value)}>
            <option value="">All Types</option>
            <option value="pyq">PYQs</option>
            <option value="lab_manual">Lab Manuals</option>
            <option value="notes">Notes</option>
          </select>
        </div>
      </div>

      {/* Documents Grid */}
      {loading ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading documents...</p>
      ) : documents.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
          <FileText size={48} color="var(--text-muted)" style={{ margin: '0 auto 1rem' }} />
          <h3>No documents found</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Upload your course PDFs or PYQs above to get started with semantic retrieval.
          </p>
        </div>
      ) : (
        <div className="doc-grid">
          {documents.map((doc) => (
            <div key={doc.id} className="doc-card">
              <div>
                <div className="doc-header">
                  <span className={`doc-badge ${doc.verification_status}`}>
                    {doc.verification_status}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                    {doc.file_type.replace('_', ' ')}
                  </span>
                </div>

                <h4 className="doc-title">{doc.title}</h4>
                <div className="doc-meta">
                  Subject: <strong>{doc.subject}</strong> | Sem: {doc.semester || 'N/A'} | Chunks: {doc.chunk_count}
                </div>
              </div>

              <div className="doc-footer">
                <div className="vote-controls">
                  <button className="vote-btn" onClick={() => handleVote(doc.id, 'up')} title="Upvote material">
                    <ThumbsUp size={14} />
                  </button>
                  <span className="score-badge" style={{ color: doc.net_score >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                    {doc.net_score}
                  </span>
                  <button className="vote-btn" onClick={() => handleVote(doc.id, 'down')} title="Downvote material">
                    <ThumbsDown size={14} />
                  </button>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <button className="btn btn-outline" style={{ padding: '0.3rem 0.6rem', fontSize: '0.8rem' }} onClick={() => openComments(doc)}>
                    <MessageSquare size={13} /> Discussion
                  </button>

                  {doc.verification_status === 'pending' && (
                    <button
                      className="btn btn-outline"
                      style={{ padding: '0.3rem 0.6rem', fontSize: '0.8rem', color: 'var(--success)' }}
                      onClick={() => handleVerify(doc.id, 'verified')}
                      title="Verify document"
                    >
                      Verify
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Comments Modal / Drawer */}
      {activeDocForComments && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100
        }}>
          <div className="card" style={{ maxWidth: '550px', width: '90%', maxHeight: '80vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>
                Discussion on: {activeDocForComments.title}
              </h3>
              <button className="btn btn-outline" style={{ padding: '0.2rem 0.5rem' }} onClick={() => setActiveDocForComments(null)}>
                X
              </button>
            </div>

            <div style={{ marginBottom: '1.5rem', maxHeight: '250px', overflowY: 'auto' }}>
              {comments.length === 0 ? (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>No comments yet. Be the first to start the discussion!</p>
              ) : (
                comments.map((c) => (
                  <div key={c.id} style={{ background: 'var(--bg)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', marginBottom: '0.5rem' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                      <span><strong>{c.user_id}</strong></span>
                      <span>{new Date(c.created_at).toLocaleTimeString()}</span>
                    </div>
                    <p style={{ fontSize: '0.9rem', marginTop: '0.25rem' }}>{c.comment_text}</p>
                  </div>
                ))
              )}
            </div>

            <form onSubmit={handleAddComment} style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="text"
                className="form-input"
                placeholder="Write an erratum note or question..."
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                required
              />
              <button type="submit" className="btn btn-primary">Post</button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

