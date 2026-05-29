import os
import re
import json
import math
import requests
import streamlit as st
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
import db
import auth
import gmail_oauth

load_dotenv()

db.create_schema()

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DIGEST_FILE        = os.getenv("DIGEST_FILE", "digest_today.json")
JSON_FILE_TODAY    = os.getenv("JSON_FILE_TODAY", "emails_today.json")
TOP_K              = 6


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _sender_name(sender: str) -> str:
    if "<" in sender:
        name = sender.split("<")[0].strip().strip('"')
        return name if name else sender
    return sender.split("@")[0]


def _sender_email(sender: str) -> str:
    if "<" in sender:
        return sender.split("<")[1].rstrip(">").strip().lower()
    return sender.strip().lower()


# ── data loading ──────────────────────────────────────────────────────────────

def _load_digest(user_id: int, gmail_account_id: int = None) -> dict | None:
    return db.get_todays_digest(user_id, gmail_account_id=gmail_account_id)


def _load_newsletters(user_id: int, gmail_account_id: int = None) -> list[dict]:
    emails = db.get_todays_emails(user_id, gmail_account_id=gmail_account_id, category="newsletter")
    for e in emails:
        if e.get("topics") is None:
            e["topics"] = []
        if e.get("images") is None:
            e["images"] = []
    return emails


# ── embedding + retrieval ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma  = math.sqrt(sum(x * x for x in a))
    mb  = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0


def _build_digest_corpus(digest) -> list[dict]:
    """Digest/overview chunks — always kept in memory (small set, ~10-20 items)."""
    chunks = []
    if not digest or "topics" not in digest:
        return chunks
    for topic, data in digest["topics"].items():
        overview = data.get("overview", "")
        if overview:
            chunks.append({
                "text":         f"[{topic}] Overview: {overview}",
                "topic":        topic,
                "headline":     f"{topic.replace('_', ' ').title()} -- overview",
                "source":       "digest",
                "source_email": "",
                "type":         "overview",
                "importance":   "high",
                "links":        [],
            })
        for section in data.get("sections", []):
            chunks.append({
                "text":         f"[{topic}] {section['headline']}: {section['summary']}",
                "topic":        topic,
                "headline":     section["headline"],
                "source":       ", ".join(section.get("sources", [])),
                "source_email": "",
                "type":         "digest_section",
                "importance":   section.get("importance", "medium"),
                "links":        [],
            })
    return chunks


def _build_raw_email_corpus(newsletters: list[dict]) -> list[dict]:
    """Raw email paragraph chunks — fallback corpus used when pgvector is not installed."""
    chunks = []
    for email in newsletters:
        full_body    = email.get("clean_body") or email.get("body", "")
        topics       = email.get("topics") or ["general_tech"]
        subject      = email.get("subject", "")
        sender_raw   = email.get("sender", "")
        sender_name  = _sender_name(sender_raw)
        sender_email_addr = _sender_email(sender_raw)
        links        = email.get("links", [])

        paras = [p.strip() for p in re.split(r"\n{2,}", full_body) if len(p.strip()) >= 80]
        if not paras:
            paras = [full_body[:700]] if full_body.strip() else []

        for para in paras[:6]:
            for topic in topics:
                chunks.append({
                    "text":         f"[{topic}] {subject}: {para[:700]}",
                    "topic":        topic,
                    "headline":     subject,
                    "source":       sender_name,
                    "source_email": sender_email_addr,
                    "type":         "raw_email",
                    "importance":   "low",
                    "links":        links,
                })
    return chunks


@st.cache_resource(show_spinner=False)
def build_knowledge_base(user_id: int | None, gmail_account_id: int | None = None):
    digest      = _load_digest(user_id, gmail_account_id=gmail_account_id)
    newsletters = _load_newsletters(user_id, gmail_account_id=gmail_account_id)

    # Attach article links to each newsletter (one bulk DB query)
    email_ids = [e["id"] for e in newsletters if e.get("id")]
    links_map = db.get_links_for_emails(email_ids) if email_ids else {}
    for e in newsletters:
        e["links"] = links_map.get(e.get("id"), [])

    # Digest/overview chunks — always in memory
    corpus = _build_digest_corpus(digest)

    # Raw email chunks — in memory only when pgvector is unavailable (fallback path)
    if not db._PGVECTOR:
        corpus += _build_raw_email_corpus(newsletters)

    if not corpus:
        return [], [], digest, newsletters
    embeddings = [_embed(c["text"]) for c in corpus]
    return corpus, embeddings, digest, newsletters


def retrieve(
    query: str,
    corpus: list[dict],
    embeddings: list[list[float]],
    topic_filter: list[str] | None = None,
    source_filter: list[str] | None = None,
    user_id: int | None = None,
    date_only: bool = True,
    gmail_account_id: int | None = None,
) -> list[dict]:
    q_emb = _embed(query)

    # In-memory cosine over corpus (digest chunks always; raw_email chunks when pgvector absent)
    scored = []
    for c, emb in zip(corpus, embeddings):
        if topic_filter and c["topic"] not in topic_filter:
            continue
        if source_filter and c["source"] not in source_filter and c["source_email"] not in source_filter:
            continue
        scored.append((c, _cosine(q_emb, emb)))
    scored.sort(key=lambda x: x[1], reverse=True)
    in_memory_top = [c for c, _ in scored[:TOP_K]]

    # pgvector ANN search for raw email chunks (when available)
    pgvector_raw: list[dict] = []
    if user_id is not None and db._PGVECTOR:
        try:
            candidates = db.search_similar_chunks(user_id, q_emb, top_k=TOP_K * 2, date_only=date_only, gmail_account_id=gmail_account_id)
            for c in candidates:
                if topic_filter and c["topic"] not in topic_filter:
                    continue
                if source_filter and c["source"] not in source_filter and c["source_email"] not in source_filter:
                    continue
                pgvector_raw.append(c)
                if len(pgvector_raw) >= TOP_K:
                    break
        except Exception:
            pass

    # Merge: if pgvector returned results, use them for raw_email; else in_memory covers everything
    if pgvector_raw:
        digest_chunks = [c for c in in_memory_top if c["type"] in ("digest_section", "overview")]
        seen = {c["text"] for c in digest_chunks}
        raw_chunks = [c for c in pgvector_raw if c["text"] not in seen]
        # Cap digest at TOP_K-2 so raw_email chunks (which carry article links) always get ≥2 slots
        combined = digest_chunks[: TOP_K - 2] + raw_chunks
    else:
        combined = in_memory_top

    combined.sort(key=lambda c: (c["type"] == "raw_email", c["importance"] != "high"))

    # Limit to 2 chunks per source so one newsletter can't flood the entire context
    source_count: dict[str, int] = {}
    deduped: list[dict] = []
    for chunk in combined:
        src = chunk.get("source", "")
        count = source_count.get(src, 0)
        # Digest overviews are always kept (they summarise across sources)
        if chunk.get("type") in ("digest_section", "overview") or count < 2:
            source_count[src] = count + 1
            deduped.append(chunk)

    return deduped[:TOP_K]


# ── answer generation ─────────────────────────────────────────────────────────

def stream_answer(
    question: str,
    chunks: list[dict],
    source_hint: str = "",
    history: list[dict] | None = None,
):
    if not chunks:
        yield "I don't have enough information to answer that. Try fetching emails and generating a digest first (Steps 1 & 2 in the sidebar)."
        return

    context = "\n\n".join(
        f"[{c['topic'].upper()}] {c['headline']}\nSource: {c['source']}\n{c['text']}"
        for c in chunks
    )
    source_instruction = (
        f"The user is specifically interested in content from: {source_hint}. "
        "Focus your answer on that source's content when available.\n\n"
        if source_hint else ""
    )

    # Build conversation memory block from the last 3 exchanges (6 messages)
    history_block = ""
    if history:
        recent = [m for m in history if m["role"] in ("user", "assistant")][-6:]
        if recent:
            lines = []
            for m in recent:
                role    = "User" if m["role"] == "user" else "Assistant"
                content = m["content"][:400].strip()
                lines.append(f"{role}: {content}")
            history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    prompt = f"""You are a helpful newsletter digest assistant. Answer the user's question using ONLY the context below.
{source_instruction}{history_block}Rules:
- Be concise and well-structured. Use bullet points for lists of stories.
- Cite newsletter source names inline (e.g. "according to Ben's Bites").
- If the conversation history shows what the user was just asking about, use that to interpret follow-up questions correctly.
- If the context doesn't contain enough information, say so honestly — do not make things up.

Context:
{context}

Question: {question}

Answer:"""

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  True,
            "options": {"temperature": 0.3, "top_p": 0.9},
        },
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                token = json.loads(line).get("response", "")
                yield token



# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Newsletter Bot",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base font (don't override icon fonts) ── */
body, p, span, h1, h2, h3, h4, h5, h6, button, input, textarea, select, label, a {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Layout ── */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; height: 0 !important; }
[data-testid="stSidebarCollapsedControl"],
button[aria-label="Close sidebar"], button[aria-label="Open sidebar"] {
    display: none !important;
}
.main .block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 780px; }

/* ── Sidebar shell ── */
section[data-testid="stSidebar"] {
    display: block !important;
    transform: none !important;
    min-width: 220px !important;
    background: #0d0f18 !important;
    border-right: 1px solid #1a1f2e !important;
    position: relative !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0.6rem 0.7rem 2rem !important;
    overflow-y: auto;
    height: 100%;
}

/* Tighten element spacing in sidebar */
section[data-testid="stSidebar"] .stElementContainer { margin-bottom: 0 !important; }
section[data-testid="stSidebar"] .stMarkdown        { margin-bottom: 0 !important; }
section[data-testid="stSidebar"] .stButton          { margin-bottom: 0 !important; }
section[data-testid="stSidebar"] hr                 { margin: 0.4rem 0 !important; border-color: #1a1f2e !important; }

/* ════════════════════════════════════════
   SIDEBAR — BUTTONS (nuclear reset then re-style)
════════════════════════════════════════ */
section[data-testid="stSidebar"] .stButton button {
    all: unset !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    box-sizing: border-box !important;
    cursor: pointer !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    transition: background 0.12s, box-shadow 0.12s !important;
}

/* Primary (New Chat, Fetch, Generate) — config.toml sets color, we add gradient */
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    padding: 0.48rem 1rem !important;
    justify-content: center !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.28) !important;
}
section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
    box-shadow: 0 4px 18px rgba(99,102,241,0.48) !important;
    filter: brightness(1.08) !important;
}

/* Secondary — nav item style (session history) */
section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    color: #8a96aa !important;
    font-weight: 400 !important;
    padding: 0.28rem 0.55rem !important;
    justify-content: flex-start !important;
    text-overflow: ellipsis !important;
    overflow: hidden !important;
    white-space: nowrap !important;
}
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.055) !important;
    color: #d0d8e8 !important;
}

/* Column buttons (Sign out, Disconnect) — outlined */
section[data-testid="stSidebar"] [data-testid="column"] .stButton button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #8a96aa !important;
    font-weight: 500 !important;
    padding: 0.35rem 0.5rem !important;
    justify-content: center !important;
    white-space: normal !important;
    font-size: 0.76rem !important;
}
section[data-testid="stSidebar"] [data-testid="column"] .stButton button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.16) !important;
    color: #d0d8e8 !important;
}

/* Download button */
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #8a96aa !important;
    padding: 0.38rem 0.5rem !important;
    justify-content: center !important;
    font-size: 0.76rem !important;
}
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #d0d8e8 !important;
}

/* ════════════════════════════════════════
   SIDEBAR — COMPONENTS
════════════════════════════════════════ */
.sidebar-brand {
    display: flex; align-items: center; gap: 0.45rem;
    padding: 0.3rem 0.1rem 0.5rem;
    margin-bottom: 0.3rem;
    border-bottom: 1px solid #1a1f2e;
}
.sidebar-brand-icon {
    font-size: 1rem;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sidebar-brand-name { font-size: 0.85rem; font-weight: 700; color: #c4c9d8; letter-spacing: -0.01em; }

.section-label {
    font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #3d4e6a;
    margin: 0.75rem 0 0.15rem 0.1rem;
    display: block;
}

.sess-time { font-size: 0.58rem; color: #3d4e6a; padding: 1px 0 3px 1.2rem; display: block; }

.user-card {
    display: flex; align-items: center; gap: 0.55rem;
    padding: 0.55rem 0.6rem;
    background: #131826; border: 1px solid #1c2235;
    border-radius: 9px; margin-bottom: 0.35rem;
}
.user-avatar {
    width: 30px; height: 30px; border-radius: 7px;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; color: #fff; flex-shrink: 0;
}
.user-name  { font-size: 0.8rem; font-weight: 600; color: #c4c9d8; line-height: 1.2; }
.user-email { font-size: 0.63rem; color: #4a5a7a; margin-top: 1px; }

.gmail-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.67rem; color: #10b981;
    background: #10b98110; border: 1px solid #10b98122;
    border-radius: 20px; padding: 2px 7px; margin-bottom: 0.4rem;
}

.step-header {
    display: flex; align-items: center; gap: 0.4rem;
    margin: 0.65rem 0 0.2rem 0;
}
.step-badge {
    width: 15px; height: 15px; border-radius: 4px;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    color: #fff; font-size: 0.55rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-title { font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #4a5a7a; }

.stat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 4px; margin: 0.35rem 0; }
.stat-box { background: #131826; border: 1px solid #1c2235; border-radius: 7px; padding: 0.4rem 0.15rem; text-align: center; }
.stat-val { font-size: 1rem; font-weight: 700; color: #a78bfa; }
.stat-lbl { font-size: 0.55rem; color: #4a5a7a; text-transform: uppercase; letter-spacing: 0.05em; }

/* ════ AUTH PAGE ════ */
/* Background orbs */
.auth-bg { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
.auth-orb-1 {
    position: absolute; border-radius: 50%;
    width: 700px; height: 700px; top: -280px; left: -200px;
    background: radial-gradient(circle, rgba(99,102,241,0.13) 0%, transparent 62%);
}
.auth-orb-2 {
    position: absolute; border-radius: 50%;
    width: 550px; height: 550px; bottom: -200px; right: -150px;
    background: radial-gradient(circle, rgba(167,139,250,0.10) 0%, transparent 62%);
}
.auth-orb-3 {
    position: absolute; border-radius: 50%;
    width: 320px; height: 320px; top: 42%; right: 12%;
    background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 62%);
}

/* Header card */
.auth-card {
    background: linear-gradient(160deg, #0d1020 0%, #090b14 100%);
    border: 1px solid #1e2438;
    border-radius: 18px;
    padding: 2.25rem 2rem 1.75rem;
    box-shadow: 0 24px 64px rgba(0,0,0,0.45), 0 0 0 1px rgba(99,102,241,0.07);
    text-align: center;
    margin-bottom: 0.75rem;
}
.auth-logo-box {
    width: 56px; height: 56px; margin: 0 auto 1.1rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #16193e 0%, #0f1130 100%);
    border: 1px solid rgba(99,102,241,0.35);
    box-shadow: 0 0 40px rgba(99,102,241,0.22), inset 0 1px 0 rgba(255,255,255,0.05);
    display: flex; align-items: center; justify-content: center;
}
.auth-logo-glyph {
    font-size: 1.6rem;
    background: linear-gradient(135deg, #6366f1 0%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.auth-title {
    font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #e2e8f0 0%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem; line-height: 1.15;
}
.auth-subtitle { font-size: 0.82rem; color: #4a5a7a; margin-bottom: 1.3rem; }
.auth-pills { display: flex; justify-content: center; gap: 0.45rem; flex-wrap: wrap; }
.auth-pill {
    font-size: 0.68rem; color: #5a6a8a;
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.16);
    border-radius: 20px; padding: 3px 11px;
}

/* Trust strip */
.auth-trust {
    display: flex; justify-content: center; gap: 1.75rem;
    margin-top: 1.5rem; flex-wrap: wrap;
}
.auth-trust-item { font-size: 0.68rem; color: #2d3a5a; }
.auth-trust-item span { color: #3a4e6a; margin-right: 4px; }

/* ════ MAIN CHAT ════ */
.chat-header-title {
    font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(135deg, #e2e8f0 20%, #a78bfa 80%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.15; letter-spacing: -0.03em; margin-bottom: 0.15rem;
}
.chat-header-sub { font-size: 0.8rem; color: #3d4e6a; margin-bottom: 1.25rem; }

[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    border: 1px solid #1a1f2e !important;
    background: #0d0f18 !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: 10px !important;
    border: 1px solid #1e2540 !important;
    background: #0d0f18 !important;
    color: #c4c9d8 !important;
}

/* ════ MISC COMPONENTS ════ */
.empty-state { text-align: center; padding: 4rem 1rem; }
.empty-state-icon { font-size: 2.8rem; margin-bottom: 0.75rem; opacity: 0.4; }
.empty-state-title { font-size: 0.95rem; font-weight: 600; color: #3d4e6a; margin-bottom: 0.4rem; }
.empty-state-body  { font-size: 0.82rem; color: #2d3a5a; line-height: 1.6; }

.source-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.source-tag { font-size: 0.66rem; color: #6366f1; background: #6366f10e; border: 1px solid #6366f120; border-radius: 4px; padding: 2px 6px; font-weight: 500; }

.pinned-banner { border-left: 2px solid #6366f1; padding: 0.5rem 0.85rem; border-radius: 0 8px 8px 0; background: #6366f108; margin-bottom: 0.9rem; font-size: 0.8rem; color: #94a3b8; }
.pinned-banner strong { color: #a78bfa; }

.read-more-bar { margin-top: 10px; padding: 8px 11px; background: #0a0c14; border: 1px solid #1a1f2e; border-radius: 8px; }
.read-more-label { font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #2d3a5a; margin-bottom: 6px; }
.read-more-links { display: flex; flex-wrap: wrap; gap: 5px; }
.read-more-link { display: inline-flex; align-items: center; gap: 4px; font-size: 0.75rem; color: #8b8ff8; background: #6366f10c; border: 1px solid #6366f122; border-radius: 6px; padding: 3px 9px; text-decoration: none; white-space: nowrap; max-width: 260px; overflow: hidden; text-overflow: ellipsis; transition: background 0.12s; }
.read-more-link:hover { background: #6366f120; color: #a78bfa; }
.read-more-link .arrow { font-size: 0.65rem; opacity: 0.5; }

.source-row { display: flex; align-items: flex-start; gap: 7px; padding: 5px 0; border-bottom: 1px solid #1a1f2e; }
.source-row:last-child { border-bottom: none; }
.source-dot { width: 5px; height: 5px; border-radius: 50%; background: #6366f1; flex-shrink: 0; margin-top: 5px; }
.source-name { font-size: 0.79rem; font-weight: 600; color: #c4c9d8; }
.source-subject { font-size: 0.69rem; color: #3d4e6a; margin-top: 2px; }

hr { border-color: #1a1f2e !important; margin: 0.5rem 0 !important; }
[data-testid="stExpander"] { border: 1px solid #1a1f2e !important; border-radius: 8px !important; }
[data-testid="stCheckbox"] label { font-size: 0.79rem !important; color: #64748b !important; }
</style>
""", unsafe_allow_html=True)

# ── resizable sidebar (JS injected into parent frame) ────────────────────────
import streamlit.components.v1 as _components
_components.html("""
<script>
(function() {
  const doc = window.parent.document;
  function init() {
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb || doc.getElementById('_sb_drag')) return;
    sb.style.position = 'relative';

    const handle = doc.createElement('div');
    handle.id = '_sb_drag';
    Object.assign(handle.style, {
      position:'absolute', right:'-3px', top:'0', bottom:'0',
      width:'6px', cursor:'col-resize', zIndex:'9999',
      borderRadius:'3px', transition:'background 0.2s'
    });
    sb.appendChild(handle);

    let drag=false, x0=0, w0=0;
    handle.addEventListener('mousedown', e=>{
      drag=true; x0=e.clientX; w0=sb.getBoundingClientRect().width;
      doc.body.style.userSelect='none'; e.preventDefault();
    });
    handle.addEventListener('mouseenter',()=>handle.style.background='rgba(99,102,241,.45)');
    handle.addEventListener('mouseleave',()=>{ if(!drag) handle.style.background=''; });
    doc.addEventListener('mousemove', e=>{
      if(!drag) return;
      const w = Math.min(520, Math.max(180, w0 + e.clientX - x0));
      sb.style.width = sb.style.minWidth = w+'px';
      const inner = sb.querySelector(':scope > div');
      if(inner) inner.style.width = w+'px';
    });
    doc.addEventListener('mouseup', ()=>{ drag=false; handle.style.background=''; doc.body.style.userSelect=''; });
  }
  if(doc.readyState==='loading') doc.addEventListener('DOMContentLoaded',init);
  else { init(); setTimeout(init, 800); }
})();
</script>
""", height=0)

# ── OAuth callback handler (runs before any session check) ───────────────────
if "code" in st.query_params and "state" in st.query_params:
    try:
        _uid, _token = gmail_oauth.exchange_code(
            st.query_params["code"],
            st.query_params["state"],
        )
        from google.oauth2.credentials import Credentials as _Creds
        _creds_obj = _Creds(
            token=_token["token"],
            refresh_token=_token["refresh_token"],
            token_uri=_token["token_uri"],
            client_id=_token["client_id"],
            client_secret=_token["client_secret"],
            scopes=_token["scopes"],
        )
        _gmail_addr = gmail_oauth.get_gmail_address(_creds_obj)
        db.save_gmail_account(_uid, _gmail_addr, refresh_token=json.dumps(_token))
        st.query_params.clear()

        # Restore user session so they land directly on the main app
        _user = db.get_user_by_id(_uid)
        if _user:
            st.session_state.user = dict(_user)
            st.rerun()
        else:
            st.success(f"Gmail connected ({_gmail_addr}). Please log in to continue.")
            st.stop()
    except Exception as _e:
        st.query_params.clear()
        st.error(f"Gmail connection failed: {_e}")
        st.stop()

# ── session state init ────────────────────────────────────────────────────────
if "user"              not in st.session_state: st.session_state.user              = None
if "messages"          not in st.session_state: st.session_state.messages          = []
if "auth_mode"         not in st.session_state: st.session_state.auth_mode         = "login"
if "fetch_done"        not in st.session_state: st.session_state.fetch_done        = False
if "fetched_count"     not in st.session_state: st.session_state.fetched_count     = 0
if "current_session_id" not in st.session_state: st.session_state.current_session_id = None


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════

def _auth_page():
    # Background decoration
    st.markdown("""
    <div class="auth-bg">
        <div class="auth-orb-1"></div>
        <div class="auth-orb-2"></div>
        <div class="auth-orb-3"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:3.5rem'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.35, 1])
    with mid:
        # Header card
        st.markdown("""
        <div class="auth-card">
            <div class="auth-logo-box">
                <span class="auth-logo-glyph">✦</span>
            </div>
            <div class="auth-title">NewsletterBot</div>
            <div class="auth-subtitle">Your personal AI-powered inbox digest</div>
            <div class="auth-pills">
                <span class="auth-pill">📬 Read-only Gmail</span>
                <span class="auth-pill">✨ Daily AI digest</span>
                <span class="auth-pill">🔒 Secure tokens</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Sign in", "Create account"])

        with tab_login:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            email    = st.text_input("Email", key="login_email", placeholder="you@example.com", label_visibility="collapsed")
            password = st.text_input("Password", key="login_password", type="password", placeholder="Password", label_visibility="collapsed")
            st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
            if st.button("Sign in →", use_container_width=True, type="primary", key="btn_login"):
                if not email or not password:
                    st.error("Please fill in both fields.")
                else:
                    try:
                        user = auth.login(email, password)
                        st.session_state.user     = user
                        st.session_state.messages = []
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        with tab_signup:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            name     = st.text_input("Full name",     key="signup_name",     placeholder="Full name",     label_visibility="collapsed")
            email    = st.text_input("Email address", key="signup_email",    placeholder="Email address", label_visibility="collapsed")
            password = st.text_input("Password",      key="signup_password", type="password", placeholder="Password (min. 6 chars)", label_visibility="collapsed")
            st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
            if st.button("Create account →", use_container_width=True, type="primary", key="btn_signup"):
                if not name or not email or not password:
                    st.error("Please fill in all fields.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        user = auth.signup(name, email, password)
                        st.session_state.user     = user
                        st.session_state.messages = []
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        # Trust strip
        st.markdown("""
        <div class="auth-trust">
            <span class="auth-trust-item"><span>✓</span> No inbox modifications</span>
            <span class="auth-trust-item"><span>✓</span> Newsletters only</span>
            <span class="auth-trust-item"><span>✓</span> Tokens never shared</span>
        </div>
        """, unsafe_allow_html=True)


if st.session_state.user is None:
    _auth_page()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# GMAIL CONNECT PAGE  (shown once, after first login)
# ══════════════════════════════════════════════════════════════════════════════

user             = st.session_state.user
user_id          = user["id"]
_gmail_account   = db.get_gmail_account(user_id)
gmail_account_id = _gmail_account["id"] if _gmail_account else None

if not gmail_oauth.is_connected(user_id):
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown('<div class="auth-logo">📬</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Connect Gmail</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-subtitle">Link your inbox so NewsletterBot can fetch and summarise your newsletters daily.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div style='background:#1e2130;border-radius:12px;padding:1rem 1.25rem;margin:1rem 0;'>
          <div style='font-size:0.82rem;color:#94a3b8;line-height:2'>
            ✅ &nbsp;Read-only access — we never modify your inbox<br>
            ✅ &nbsp;Only newsletter emails are processed<br>
            ✅ &nbsp;Tokens stored securely, never shared
          </div>
        </div>
        """, unsafe_allow_html=True)
        auth_url = gmail_oauth.get_auth_url(user_id)
        st.link_button("Connect Gmail via Google →", url=auth_url, use_container_width=True, type="primary")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to sign in", use_container_width=True):
            st.session_state.user = None
            st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

corpus, embeddings, digest, newsletters = build_knowledge_base(user_id, gmail_account_id=gmail_account_id)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── brand ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-brand">'
        '<span class="sidebar-brand-icon">✦</span>'
        '<span class="sidebar-brand-name">NewsletterBot</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── new chat button ───────────────────────────────────────────────────────
    if st.button("+ New Chat", use_container_width=True, type="primary", key="btn_new_chat"):
        st.session_state.messages           = []
        st.session_state.current_session_id = None
        st.session_state.pop("pinned_email", None)
        st.session_state.pop("pinned_label", None)
        st.rerun()

    # ── recent sessions ───────────────────────────────────────────────────────
    recent_sessions = db.get_recent_sessions(user_id, gmail_account_id=gmail_account_id, limit=10)
    if recent_sessions:
        st.markdown('<p class="section-label">Recent</p>', unsafe_allow_html=True)
        for sess in recent_sessions:
            sid     = sess["id"]
            title   = sess["title"] or "New Chat"
            updated = sess.get("updated_at")
            if updated:
                now  = datetime.now()
                diff = now - updated.replace(tzinfo=None) if updated.tzinfo else now - updated
                time_label = updated.strftime("%H:%M") if diff.days == 0 else (
                    "Yesterday" if diff.days == 1 else updated.strftime("%b %d")
                )
            else:
                time_label = ""

            is_active = (st.session_state.current_session_id == sid)
            icon  = "▸" if is_active else "·"
            short = title[:26] + ("…" if len(title) > 26 else "")
            label = f"{icon}  {short}"
            if st.button(label, key=f"sess_{sid}", use_container_width=True, help=title):
                msgs = db.get_session_messages(sid)
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in msgs
                ]
                st.session_state.current_session_id = sid
                st.rerun()
            if time_label:
                st.markdown(f'<span class="sess-time">{time_label}</span>', unsafe_allow_html=True)

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    st.divider()

    # ── user card ─────────────────────────────────────────────────────────────
    initials   = "".join(w[0].upper() for w in user["name"].split()[:2])
    gmail_addr = _gmail_account.get("gmail_address", "") if _gmail_account else ""
    st.markdown(
        f'<div class="user-card">'
        f'<div class="user-avatar">{initials}</div>'
        f'<div><div class="user-name">{user["name"]}</div>'
        f'<div class="user-email">{user["email"]}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if gmail_addr:
        st.markdown(
            f'<div class="gmail-badge">● &thinsp;{gmail_addr}</div>',
            unsafe_allow_html=True,
        )

    col_lo, col_dc = st.columns(2)
    with col_lo:
        if st.button("Sign out", use_container_width=True):
            st.session_state.user               = None
            st.session_state.messages           = []
            st.session_state.current_session_id = None
            st.cache_resource.clear()
            st.rerun()
    with col_dc:
        if st.button("Disconnect", use_container_width=True):
            with db.get_cursor() as cur:
                if gmail_account_id:
                    cur.execute(
                        "UPDATE gmail_accounts SET is_active = FALSE, refresh_token = NULL WHERE id = %s",
                        (gmail_account_id,),
                    )
                else:
                    cur.execute(
                        "UPDATE gmail_accounts SET is_active = FALSE, refresh_token = NULL WHERE user_id = %s",
                        (user_id,),
                    )
            st.cache_resource.clear()
            st.rerun()

    st.divider()

    # Step 1 — Fetch emails
    st.markdown('<div class="step-header"><div class="step-badge">1</div><div class="step-title">Fetch Today\'s Emails</div></div>', unsafe_allow_html=True)
    if st.button("Fetch Today's Emails", use_container_width=True, type="primary"):
        log_area  = st.empty()
        log_lines = []

        def _log(msg):
            log_lines.append(msg)
            log_area.code("\n".join(log_lines[-12:]))

        with st.spinner("Fetching from Gmail..."):
            stats = gmail_oauth.run_fetch_pipeline(user_id, progress_cb=_log)

        if "error" in stats:
            st.error(stats["error"])
        else:
            st.session_state.fetch_done    = True
            st.session_state.fetched_count = stats.get("saved", 0)
            st.success(
                f"{stats['saved']} new  •  {stats['skipped']} skipped"
                + (f"  •  {stats['errors']} errors" if stats["errors"] else "")
            )
            st.cache_resource.clear()
            st.rerun()

    # Step 2 — Generate digest
    st.markdown('<div class="step-header"><div class="step-badge">2</div><div class="step-title">Generate Digest</div></div>', unsafe_allow_html=True)

    if not st.session_state.fetch_done:
        n_today = 0
        st.caption("Fetch emails first (Step 1).")
    else:
        todays_newsletters = db.get_todays_emails(user_id, gmail_account_id=gmail_account_id, category="newsletter")
        n_today = len(todays_newsletters)
        if n_today == 0:
            st.caption("No newsletters found in your inbox today.")
        else:
            st.caption(f"{n_today} newsletter(s) ready to summarise.")

    if st.button(
        "Generate Digest",
        use_container_width=True,
        type="primary",
        disabled=(n_today == 0),
    ):
        log_area  = st.empty()
        log_lines = []

        def _dlog(msg):
            log_lines.append(msg)
            log_area.code("\n".join(log_lines[-14:]))

        # Monkey-patch print so summarizer progress appears in the log
        import builtins, summarizer as _summ
        _orig_print = builtins.print

        def _patched_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args).strip()
            if msg:
                _dlog(msg)
            _orig_print(*args, **kwargs)

        builtins.print = _patched_print
        try:
            with st.spinner("Generating digest..."):
                digest_result = _summ.generate_daily_digest(user_id, gmail_account_id=gmail_account_id)
        finally:
            builtins.print = _orig_print

        if digest_result:
            st.success(f"Digest ready — {len(digest_result.get('topics', {}))} topic(s)")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("Digest generation failed — check that newsletters were fetched.")

    st.divider()

    # ── digest status ─────────────────────────────────────────────────────────
    if digest:
        n_topics   = len(digest.get("topics", {}))
        n_stories  = sum(len(d.get("sections", [])) for d in digest.get("topics", {}).values())
        n_sources  = len({
            src
            for d in digest.get("topics", {}).values()
            for s in d.get("sections", [])
            for src in s.get("sources", [])
        })
        gen_time = digest.get("generated_at", "")[:16].replace("T", " ")
        st.markdown(
            f"""<div class="stat-grid">
              <div class="stat-box"><div class="stat-val">{n_topics}</div><div class="stat-lbl">Topics</div></div>
              <div class="stat-box"><div class="stat-val">{n_stories}</div><div class="stat-lbl">Stories</div></div>
              <div class="stat-box"><div class="stat-val">{n_sources}</div><div class="stat-lbl">Sources</div></div>
            </div>
            <div style='font-size:0.62rem;color:#2d3a5a;margin-top:4px'>Updated {gen_time}</div>""",
            unsafe_allow_html=True,
        )

        # Download digest as markdown
        digest_lines = [f"# Newsletter Digest — {digest['date']}", ""]
        for _topic, _data in digest.get("topics", {}).items():
            digest_lines.append(f"## {_topic.replace('_', ' ').title()}")
            if _data.get("overview"):
                digest_lines += [_data["overview"], ""]
            for _s in _data.get("sections", []):
                digest_lines.append(f"### {_s['headline']}")
                if _s.get("summary"):
                    digest_lines.append(_s["summary"])
                if _s.get("sources"):
                    digest_lines.append(f"*Sources: {', '.join(_s['sources'])}*")
                digest_lines.append("")
        st.download_button(
            "Download Digest",
            data="\n".join(digest_lines),
            file_name=f"digest_{digest['date']}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.markdown(
            "<div style='font-size:0.75rem;color:#2d3a5a;padding:0.4rem 0'>No digest yet — fetch emails and generate.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── newsletter filter ─────────────────────────────────────────────────────
    all_sources = sorted({_sender_name(e.get("sender", "")) for e in newsletters if e.get("sender")})

    source_filter = None
    if all_sources:
        st.markdown('<p class="section-label">Filter by Newsletter</p>', unsafe_allow_html=True)
        selected_sources = st.multiselect(
            "newsletter_filter",
            options=all_sources,
            default=[],
            placeholder="All newsletters",
            label_visibility="collapsed",
        )
        if selected_sources:
            source_filter = selected_sources
            st.caption(f"Showing: {', '.join(selected_sources)}")

    # ── topic filter ──────────────────────────────────────────────────────────
    available_topics = sorted({
        t
        for e in newsletters
        for t in (e.get("topics") or ["general_tech"])
    } | {c["topic"] for c in corpus})
    if available_topics:
        st.markdown('<p class="section-label">Filter by Topic</p>', unsafe_allow_html=True)
        selected_topics = [
            t for t in available_topics
            if st.checkbox(t.replace("_", " ").title(), value=True, key=f"cb_{t}")
        ]
        topic_filter = selected_topics if selected_topics else available_topics
    else:
        topic_filter = None

    # ── search scope ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Search Scope</p>', unsafe_allow_html=True)
    search_all_history = st.toggle(
        "Search all history (not just today)",
        value=False,
        help="When ON, answers can draw from emails across all past days, not just today's fetch.",
    )
    date_only = not search_all_history
    if search_all_history:
        st.caption("Searching across all stored emails.")

    st.divider()

    # ── pin email from multi-sender ───────────────────────────────────────────
    sender_map: dict[str, list[dict]] = defaultdict(list)
    for email in newsletters:
        sender_map[_sender_name(email.get("sender", ""))].append(email)
    multi_senders = {k: v for k, v in sender_map.items() if len(v) > 1}

    if multi_senders:
        st.markdown('<p class="section-label">Pin Email as Context</p>', unsafe_allow_html=True)
        for sender, emails in multi_senders.items():
            with st.expander(f"{sender}  ({len(emails)})"):
                for i, e in enumerate(emails):
                    subj    = e.get("subject", "No subject")[:52]
                    preview = re.sub(r"\s+", " ", e.get("body", ""))[:80].strip()
                    if st.button(subj, key=f"pin_{sender}_{i}", use_container_width=True):
                        st.session_state["pinned_email"] = e
                        st.session_state["pinned_label"] = f"{sender} — {subj}"
                    st.caption(preview + "...")

        if st.session_state.get("pinned_email"):
            st.info(f"📌 Pinned: {st.session_state['pinned_label'][:48]}")
            if st.button("Unpin", use_container_width=True):
                st.session_state.pop("pinned_email", None)
                st.session_state.pop("pinned_label", None)
                st.rerun()
        st.divider()

    # ── today's digest overview ───────────────────────────────────────────────
    if digest and digest.get("topics"):
        st.markdown('<p class="section-label">Today\'s Digest</p>', unsafe_allow_html=True)
        imp_dot = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for topic, data in digest["topics"].items():
            sections  = data.get("sections", [])
            label     = f"{topic.replace('_', ' ').title()}  ({len(sections)})"
            with st.expander(label):
                overview = data.get("overview", "")
                if overview:
                    st.caption(overview)
                for s in sections:
                    imp = s.get("importance", "medium")
                    st.markdown(
                        f"{imp_dot.get(imp, '⚪')} **{s['headline'][:70]}**",
                        unsafe_allow_html=False,
                    )

    st.divider()
    _cc, = st.columns([1])
    with _cc:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages           = []
            st.session_state.current_session_id = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT
# ══════════════════════════════════════════════════════════════════════════════

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div class='chat-header-title'>What's in the news?</div>"
    f"<div class='chat-header-sub'>{datetime.now().strftime('%A, %B %d')} &nbsp;·&nbsp; Ask me anything about your newsletters</div>",
    unsafe_allow_html=True,
)

# ── empty state ───────────────────────────────────────────────────────────────
if not corpus:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-state-icon">📭</div>
      <div class="empty-state-title">Your digest is empty</div>
      <div class="empty-state-body">
        Fetch today's emails with <strong>Step 1</strong> in the sidebar,<br>
        then run <strong>Step 2</strong> to generate your digest.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── pinned email banner ───────────────────────────────────────────────────────
pinned = st.session_state.get("pinned_email")
if pinned:
    pinned_preview = re.sub(r"\s+", " ", pinned.get("clean_body") or pinned.get("body", ""))[:180].strip()
    st.markdown(
        f'<div class="pinned-banner">'
        f'<strong>📌 Pinned context:</strong> {st.session_state.get("pinned_label", "")}<br>'
        f'<span style="color:#777;font-size:0.8rem">{pinned_preview}…</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            tags = "".join(f'<span class="source-tag">{s.strip()}</span>' for s in msg["sources"].split(",") if s.strip())
            st.markdown(f'<div class="source-tags">{tags}</div>', unsafe_allow_html=True)

# ── chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("e.g. What did GenAI Works cover today?"):
    # Create a new DB session on the first message of a conversation
    is_first_message = (st.session_state.current_session_id is None)
    if is_first_message:
        sid = db.create_chat_session(user_id, title=user_input[:80], gmail_account_id=gmail_account_id)
        st.session_state.current_session_id = sid
    else:
        sid = st.session_state.current_session_id

    st.session_state.messages.append({"role": "user", "content": user_input})
    db.append_chat_message(sid, "user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        # Build pinned context chunks
        pinned_chunks = []
        source_hint   = ""
        if pinned:
            pinned_name = _sender_name(pinned.get("sender", ""))
            source_hint = pinned_name
            pinned_text = pinned.get("clean_body") or pinned.get("body", "")
            paras = [p.strip() for p in re.split(r"\n{2,}", pinned_text) if len(p.strip()) >= 80]
            for para in paras[:10]:
                pinned_chunks.append({
                    "text":         para[:700],
                    "topic":        "pinned",
                    "headline":     pinned.get("subject", ""),
                    "source":       pinned_name,
                    "source_email": _sender_email(pinned.get("sender", "")),
                    "type":         "pinned",
                    "importance":   "high",
                })

        # When searching all history, don't restrict topics to only today's — past emails may have different topics
        effective_topic_filter = None if not date_only else topic_filter
        retrieved    = retrieve(user_input, corpus, embeddings, topic_filter=effective_topic_filter, source_filter=source_filter, user_id=user_id, date_only=date_only, gmail_account_id=gmail_account_id)
        seen         = {c["text"] for c in pinned_chunks}
        final_chunks = pinned_chunks + [c for c in retrieved if c["text"] not in seen]

        # Stream answer — pass conversation history for memory
        response_text = st.write_stream(
            stream_answer(
                user_input,
                final_chunks,
                source_hint=source_hint,
                history=st.session_state.messages[:-1],
            )
        )

        # Source tags
        sources = sorted({
            c["source"] for c in final_chunks
            if c["source"] and c["source"] not in ("digest", "")
        })
        if sources:
            tags = "".join(f'<span class="source-tag">{s}</span>' for s in sources)
            st.markdown(
                f'<div class="source-tags">{tags}</div>',
                unsafe_allow_html=True,
            )

        # ── Option D: "Read More" — article links if available, source list as fallback ──
        raw_with_links = [
            c for c in final_chunks
            if c.get("type") == "raw_email" and c.get("links")
        ]

        if raw_with_links:
            # PATH A — collect all unique article links from relevant raw chunks
            seen_urls: set[str] = set()
            link_items: list[dict] = []
            for chunk in raw_with_links:
                for lnk in chunk.get("links", []):
                    url = lnk.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        anchor = (lnk.get("text") or lnk.get("anchor_text") or "").strip()
                        label  = anchor[:55] if anchor else chunk.get("headline", "Read article")[:55]
                        link_items.append({"url": url, "label": label})
                        if len(link_items) >= 5:
                            break
                if len(link_items) >= 5:
                    break

            if link_items:
                link_tags_html = "".join(
                    f'<a class="read-more-link" href="{li["url"]}" target="_blank" rel="noopener">'
                    f'<span class="arrow">↗</span>{li["label"]}'
                    f'</a>'
                    for li in link_items
                )
                st.markdown(
                    f'<div class="read-more-bar">'
                    f'<div class="read-more-label">Read the full stories</div>'
                    f'<div class="read-more-links">{link_tags_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        else:
            # PATH B — no links stored yet; show which newsletters covered this topic
            seen_sources: set[str] = set()
            source_rows: list[dict] = []
            for chunk in final_chunks:
                sname = chunk.get("source", "").strip()
                subj  = chunk.get("headline", "").strip()
                if sname and sname not in ("digest", "") and sname not in seen_sources:
                    seen_sources.add(sname)
                    source_rows.append({"name": sname, "subject": subj})
                    if len(source_rows) >= 4:
                        break

            if source_rows:
                rows_html = "".join(
                    f'<div class="source-row">'
                    f'<div class="source-dot"></div>'
                    f'<div>'
                    f'<div class="source-name">{r["name"]}</div>'
                    + (f'<div class="source-subject">{r["subject"][:70]}</div>' if r["subject"] else "")
                    + f'</div></div>'
                    for r in source_rows
                )
                st.markdown(
                    f'<div class="read-more-bar">'
                    f'<div class="read-more-label">Covered by</div>'
                    f'{rows_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    assistant_entry = {
        "role":    "assistant",
        "content": response_text,
        "sources": ", ".join(sources) if sources else "",
    }
    st.session_state.messages.append(assistant_entry)
    db.append_chat_message(sid, "assistant", response_text)
