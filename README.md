# NewsletterBot

An AI-powered newsletter digest application that connects to your Gmail inbox, automatically classifies incoming emails, and generates daily AI summaries using a locally-running LLM (Ollama). Includes a RAG-powered chat interface to ask questions about your newsletters, persistent chat history, and a background scheduler that runs the full pipeline automatically every few hours.

Built with **Streamlit**, **PostgreSQL**, **pgvector**, and **Ollama**.

---

## Features

- **Gmail OAuth 2.0** — read-only access to your inbox; no passwords stored, tokens saved securely in PostgreSQL
- **Email classification** — every incoming email is automatically categorised (newsletter, spam, promotional, personal, etc.) using an LLM
- **Topic tagging** — newsletters are tagged across multiple topics (AI, fintech, cybersecurity, health, etc.) using semantic embeddings + LLM
- **Daily digest** — related stories from different newsletters are clustered together and synthesised into a clean multi-topic digest
- **RAG-powered chat** — ask natural language questions about your newsletters; answers are grounded in your digest using pgvector similarity search
- **Chat history** — every conversation is saved to the database and shown in the sidebar, scoped per Gmail inbox
- **Background scheduler** — `scheduler.py` runs automatically every 5 hours, fetching new emails and regenerating the digest without any manual action
- **Multi-account support** — one app account can connect multiple Gmail inboxes; each inbox has completely separate emails, digests, and chat history

---

## How It Works

```
Gmail Inbox
    │
    ▼
gmail_oauth.py        ← Fetches emails via Gmail REST API (OAuth 2.0)
    │
    ▼
classification.py     ← Classifies each email (newsletter / spam / etc.)
news_letter_classifier.py  ← Tags newsletters with topics (AI, fintech, etc.)
cleaner.py            ← Strips boilerplate, extracts clean body + article links
    │
    ▼
summarizer.py         ← Deduplicates → clusters → synthesises digest via LLM
    │
    ▼
PostgreSQL + pgvector ← Stores emails, embeddings, digest, chat sessions
    │
    ▼
bot.py (Streamlit)    ← Chat UI + sidebar controls
```

**`scheduler.py`** wraps the entire pipeline (fetch → classify → digest) and runs it automatically on a timer so you always have a fresh digest without clicking anything.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Database | PostgreSQL + pgvector |
| LLM / Embeddings | Ollama (runs locally) |
| Gmail Integration | Google OAuth 2.0 + Gmail REST API |
| Authentication | bcrypt password hashing |
| Scheduler | Python `schedule` library |

---

## Project Structure

```
newsletter-digest-bot/
├── bot.py                     # Streamlit UI — chat, sidebar, auth pages
├── db.py                      # PostgreSQL schema + all database functions
├── auth.py                    # User signup / login (bcrypt)
├── gmail_oauth.py             # Gmail OAuth flow + email fetching pipeline
├── classification.py          # LLM-based email category classifier
├── news_letter_classifier.py  # Newsletter topic tagger (embeddings + LLM)
├── summarizer.py              # Digest generation — dedup, cluster, synthesise
├── cleaner.py                 # Email body cleaner + article link extractor
├── scheduler.py               # Background scheduler — runs pipeline every N hours
├── migrate_json_to_pg.py      # One-time migration utility (JSON → PostgreSQL)
├── tests/
│   └── test_pipeline.py       # Unit tests for pure functions (no DB/Ollama needed)
├── .streamlit/
│   └── config.toml            # Dark theme configuration
├── .env.example               # Environment variable template
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ (pgvector extension optional but recommended for faster search)
- [Ollama](https://ollama.com) installed and running locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Newsletter-Digest-Bot.git
cd Newsletter-Digest-Bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull required Ollama models

```bash
ollama pull qwen2.5:14b       # digest summarisation
ollama pull qwen2.5:3b        # chat answers
ollama pull nomic-embed-text  # semantic embeddings
ollama pull mistral           # email / topic classification
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
# Google OAuth 2.0 — create credentials at console.cloud.google.com
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
OAUTH_REDIRECT_URI=http://localhost:8501

# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=newsletter_bot
DB_USER=postgres
DB_PASSWORD=your_password

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_SUMMARIZE_MODEL=qwen2.5:14b
OLLAMA_CHAT_MODEL=qwen2.5:3b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CLASSIFY_MODEL=mistral
```

### 5. Run the app

```bash
streamlit run bot.py
```

### 6. (Optional) Run the background scheduler

The scheduler automatically fetches emails and regenerates the digest every 5 hours. Run it in a separate terminal:

```bash
python scheduler.py
```

To run it once immediately and exit (useful for testing):

```bash
python scheduler.py --now
```

To change the interval, set `REPEAT_HOURS` in your `.env`:

```env
REPEAT_HOURS=3
```

---

## Usage

1. **Sign up** — create an account with your name, email, and password
2. **Connect Gmail** — authorise read-only Gmail access via the Google OAuth consent screen
3. **Fetch emails** — click **Fetch Today's Emails** in the sidebar to pull today's newsletters
4. **Generate Digest** — click **Generate Digest** to cluster and summarise the fetched emails
5. **Chat** — ask anything about your newsletters in the chat window (e.g. *"What did Ben's Bites cover today?"*)
6. **History** — previous conversations are saved in the sidebar; click any session to reload it

---

## Running Tests

Unit tests cover pure functions (no database or Ollama connection required):

```bash
pytest tests/ -v
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_SUMMARIZE_MODEL` | `qwen2.5:14b` | Model for digest synthesis |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:3b` | Model for chat answers |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_CLASSIFY_MODEL` | `mistral` | Email/topic classification model |
| `REPEAT_HOURS` | `5` | Scheduler run interval in hours |
| `DEDUP_THRESHOLD` | `0.92` | Cosine similarity threshold for deduplication |
| `MAX_CLUSTERS` | `7` | Max story clusters per topic in digest |
| `MIN_SENDER_HISTORY` | `10` | Sender history size before skipping LLM classification |
