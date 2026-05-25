# Embedding-Based Semantic Search (RAG) Dashboard

An interactive, dark-mode glassmorphism dashboard demonstrating **dense neural semantic search and Retrieval-Augmented Generation (RAG)** compared side-by-side with classical lexical keyword indexing (**TF-IDF**). 

This system represents the core organizational search technologies deployed by major enterprise platforms like Notion, Slack, and LinkedIn to surface relevant knowledge across wikis, conversations, and professional profiles.

---

## ✨ Features

- **🔍 Dual-Mode Search Comparison**: Compare classical **TF-IDF keyword matching** directly with **dense vector embeddings** (calculated via a neural sentence-transformer model and retrieved using cosine similarity). Synonyms and concept overlaps are highlighted visually.
- **🌌 2D Feature Space Map**: An interactive HTML5 Canvas visualizer mapping multi-dimensional document vector embeddings onto a 2D grid using **Principal Component Analysis (PCA)**. Supports mouse drag to pan, wheel zoom, dot hover, and draws neural pathways from your search query vector to its top nearest neighbors.
- **⚡ Step-by-Step RAG Pipeline**: An educational dashboard simulating a complete RAG system:
  1. *Retrieve*: Surface the top-3 most similar document chunks.
  2. *Inject*: Inspect the structured system instruction prompt containing the injected contexts.
  3. *Synthesize*: View the synthesized answer generated with an organic word-by-word typewriter effect.
- **📊 Real-time User Simulation & Benchmarks**: Run an automated evaluation benchmark simulating 8 synonym-shifting query scenarios. Measures and charts **Mean Reciprocal Rank (MRR)** and **Precision@K** using Chart.js, demonstrating a **+73.1% search quality improvement** of dense vectors over TF-IDF.

---

## 🏗️ Technical Architecture

- **Backend**: built with **Python 3.13** and **FastAPI**.
- **Vector Search Engine**: uses HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) generating 384-dimensional vectors, indexed using `FAISS` (IndexFlatIP).
- **Offline SVD Fallback**: If PyTorch or internet connections are constrained, the backend gracefully falls back to an offline **Latent Semantic Analysis (LSA)** vector space (TF-IDF + `scikit-learn`'s `TruncatedSVD`), keeping the vector visualizer and semantic matching completely functional.
- **Frontend**: Single-page application built using semantic **HTML5**, custom responsive **Vanilla CSS (Glassmorphism design)**, and high-performance **Vanilla JavaScript** (without complex build steps or node modules).

---

## 📁 Repository Structure

```text
├── app/
│   ├── main.py              # FastAPI server, pre-seeded corpus, evaluation & ML pipelines
│   ├── templates/
│   │   └── index.html       # Single-page dashboard markup
│   └── static/
│       ├── css/
│       │   └── style.css    # Premium HSL slate-violet dark mode and layout styles
│       └── js/
│           └── app.js       # SPA controller, 2D Canvas engine, & API integration
├── requirements.txt         # Required Python packages
├── .gitignore               # Standard Python & HuggingFace ignore patterns
└── README.md                # This file
```

---

## 🚀 How to Run the Application

### 1. Prerequisites
- Python 3.10 or higher.
- Active internet connection (only on first startup to download the 90MB MiniLM model; subsequent startups are completely cached/offline).

### 2. Set Up Virtual Environment
Create and activate your virtual environment:
```powershell
# Create venv
python -m venv venv

# Activate venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate venv (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
Install the required machine learning and web server packages:
```bash
pip install -r requirements.txt
```

### 4. Start the Server
Launch the FastAPI application using Uvicorn:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Access the Web Dashboard
Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧪 Simulation Scenarios Evaluated

The benchmark console tests retrieval against shifts in vocabulary where keyword models traditionally break down:
1. *Intent*: Vacation policy -> *Query*: "how many days can i take off?" (Matches synonym PTO)
2. *Intent*: Meal expense limits -> *Query*: "food billing budget and travel receipt limits" (Matches Concur)
3. *Intent*: Outage support -> *Query*: "who to page for p0 outage?" (Matches PagerDuty & On-Call)
4. *Intent*: Password MFA setup -> *Query*: "2fa requirements for logging in" (Matches Multi-Factor Authentication)
5. *Intent*: PR approvals -> *Query*: "code quality checks before ship" (Matches pull requests & merging)
6. *Intent*: Remote hours -> *Query*: "telecommute and core attendance times" (Matches hybrid work)
7. *Intent*: Search indexing milestones -> *Query*: "low latency indexing search project" (Matches latency & FAISS goals)
8. *Intent*: Resume parsing -> *Query*: "expert in vector databases and semantic retrieval profiles" (Matches RAG engineer Omkar Kadam)
