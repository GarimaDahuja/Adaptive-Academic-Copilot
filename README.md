# 🎓 Adaptive Academic Copilot

An AI-powered study assistant for Indian university students (Semester 3–5 CSE focus). Combines semantic document retrieval, Gemini-powered Q&A, reinforcement-learning-based curriculum recommendations, and spaced-repetition quizzes into one full-stack application.

---

## ✨ Features

| Feature | Description |
|---|---|
| **📄 Document Hub** | Upload PDFs (PYQs, lab manuals, notes). Community voting + peer verification. |
| **🤖 Ask AI** | Gemini-grounded Q&A with page-level citations from your uploaded documents. |
| **🧠 RL Recommendations** | LinUCB contextual bandit + Q-learning/SARSA adaptively suggests what to study next. |
| **📊 Mastery Dashboard** | Per-topic mastery scores (0–1) updated via exponential smoothing after each quiz. |
| **🃏 Quizzes** | Interactive quizzes with spaced repetition scheduling (1/3/7-day intervals). |

---

## 🏗️ Architecture

```
Adaptive Academic Copilot/
├── backend/                  # Flask REST API
│   ├── app/
│   │   ├── __init__.py       # Application factory
│   │   ├── config.py         # Dev / Test configuration
│   │   ├── database.py       # SQLAlchemy db instance
│   │   ├── models/           # SQLAlchemy models
│   │   │   ├── document.py   # Document, DocumentChunk, Comment
│   │   │   └── quiz.py       # QuizQuestion, StudentMastery, QuizAttempt
│   │   ├── routes/           # Flask Blueprints
│   │   │   ├── health.py     # GET /api/health
│   │   │   ├── documents.py  # Document CRUD + voting
│   │   │   ├── qa.py         # Gemini Q&A + embeddings
│   │   │   └── learning.py   # Quizzes, mastery, recommendations
│   │   └── services/
│   │       ├── pdf_service.py     # PDF text extraction + chunking
│   │       ├── gemini_service.py  # Gemini API, embeddings, RAG
│   │       └── mastery_service.py # Mastery updates + spaced repetition
│   ├── tests/                # 48 pytest tests (in-memory SQLite)
│   └── run.py                # Flask entry point
├── rl/                       # Isolated RL module (Gymnasium)
│   ├── envs/student_env.py   # StudentEnv — 5-topic Gymnasium environment
│   ├── policies/             # Random, Linear, LinUCB, Q-Learning, SARSA
│   ├── evaluation/evaluate.py
│   ├── service.py            # AdaptiveRecommendationService (JSON checkpoints)
│   └── tests/test_rl.py
├── frontend/                 # React + Vite SPA
│   └── src/
│       ├── components/
│       │   ├── Navbar.jsx
│       │   ├── DocumentManager.jsx
│       │   ├── AskAI.jsx
│       │   ├── MasteryDashboard.jsx
│       │   └── QuizPlayer.jsx
│       ├── api.js            # Axios API client
│       ├── App.jsx
│       └── main.jsx
├── docker-compose.yml        # pgvector PostgreSQL (port 5432)
├── requirements.txt
└── pytest.ini
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker Desktop (only for the live PostgreSQL database; tests use SQLite)

### 1. Clone & Install

```bash
git clone https://github.com/GarimaDahuja/Adaptive-Academic-Copilot.git
cd Adaptive-Academic-Copilot
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY
```

```env
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://copilot:copilot123@localhost:5432/adaptive_copilot
GEMINI_API_KEY=your-gemini-api-key
```

### 3. Start the Database (Docker)

```bash
docker compose up -d
```

### 4. Start the Backend

```bash
python backend/run.py
# → http://127.0.0.1:5000
```

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## 🧪 Running Tests

Tests use in-memory SQLite — **no Docker required**.

```bash
# From project root
python -m pytest

# Verbose
python -m pytest -v

# RL module only
python -m pytest rl/tests/

# Backend only
python -m pytest backend/tests/
```

**Current status: 48 / 48 passing ✅**

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Health check |
| `POST` | `/api/documents/upload` | Upload PDF |
| `GET`  | `/api/documents` | List documents |
| `POST` | `/api/documents/<id>/vote` | Upvote / downvote |
| `POST` | `/api/documents/<id>/verify` | Mark as verified |
| `GET`/`POST` | `/api/documents/<id>/comments` | Comments |
| `POST` | `/api/qa/ask` | Gemini Q&A with citations |
| `POST` | `/api/qa/embed-document/<id>` | Generate embeddings |
| `POST` | `/api/learning/quizzes` | Create quiz question |
| `GET`  | `/api/learning/quizzes` | List quizzes |
| `POST` | `/api/learning/quizzes/<id>/submit` | Submit answer, update mastery |
| `GET`  | `/api/learning/mastery/<student_id>` | Get mastery profile |
| `GET`  | `/api/learning/recommendations/<student_id>` | RL topic recommendation |

---

## 🤖 RL Module

The `rl/` directory is a **standalone Gymnasium environment** independent of Flask.

### Policies (5 total)

| Policy | Type | Algorithm |
|--------|------|-----------|
| `RandomPolicy` | Baseline | Uniform random action |
| `LinearCurriculumPolicy` | Baseline | Fixed topic ordering |
| `ContextualBanditPolicy` | Online | LinUCB (linear upper confidence bound) |
| `QLearningPolicy` | Offline TD | ε-greedy + Bellman update |
| `SARSAPolicy` | Online TD | On-policy ε-greedy |

### Benchmark Results (500 episodes)

| Policy | Avg Reward |
|--------|-----------|
| Random | 4.44 |
| Contextual Bandit | 6.74 |
| Q-Learning | 15.14 |
| SARSA | 15.81 |
| Linear Curriculum | 17.90 |

### Run the benchmark

```bash
python rl/evaluation/evaluate.py
```

---

## 📐 Spaced Repetition Schedule

After each quiz attempt the student's mastery score is updated via **exponential smoothing**:

```
new_mastery = α × score + (1 − α) × old_mastery    (α = 0.3)
```

The next review is scheduled based on the resulting mastery level:

| Mastery | Next Review |
|---------|-------------|
| < 0.40  | 1 day       |
| 0.40 – 0.70 | 3 days  |
| ≥ 0.70  | 7 days      |

---

## 🛠️ Tech Stack

**Backend**: Flask · SQLAlchemy · pgvector · pypdf · Google Gemini API  
**RL**: Gymnasium · NumPy  
**Frontend**: React 18 · Vite · Axios  
**Database**: PostgreSQL 16 + pgvector (Docker) / SQLite (tests)  
**Testing**: pytest · 48 tests  

---

## 📄 License

MIT
