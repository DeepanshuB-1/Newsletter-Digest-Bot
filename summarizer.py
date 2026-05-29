import os
import re
import math
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import db

load_dotenv()

OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_SUMMARIZE_MODEL = os.getenv("OLLAMA_SUMMARIZE_MODEL", "qwen2.5:14b")
OLLAMA_EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Tuned defaults (override via .env):
# DEDUP_THRESHOLD  0.88→0.92: only drop near-identical paragraphs, not just similar ones
# CLUSTER_THRESHOLD 0.72→0.76: tighter clusters so loosely related stories stay separate
# MAX_CLUSTERS     5→7: surface more distinct stories per topic
DEDUP_THRESHOLD   = float(os.getenv("DEDUP_THRESHOLD",   "0.92"))
CLUSTER_THRESHOLD = float(os.getenv("CLUSTER_THRESHOLD", "0.76"))
MAX_CLUSTERS      = int(os.getenv("MAX_CLUSTERS",        "7"))
EMBED_WORKERS     = int(os.getenv("EMBED_WORKERS",       "6"))


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_todays_newsletters(user_id: int, gmail_account_id: int = None) -> list[dict]:
    """Load today's newsletters from DB for the given user's inbox."""
    emails = db.get_todays_emails(user_id, gmail_account_id=gmail_account_id, category="newsletter")
    for e in emails:
        if e.get("topics") is None:
            e["topics"] = []
    return emails


def _chunk_body(body: str, chunk_size: int = 700, min_len: int = 80) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", body) if len(p.strip()) >= min_len]
    if not paras:
        paras = [p.strip() for p in body.splitlines() if len(p.strip()) >= min_len]
    if not paras:
        return [body[:chunk_size]] if body.strip() else []

    chunks, current = [], ""
    for para in paras:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para[:chunk_size]
    if current:
        chunks.append(current)
    return chunks


def _expand_newsletters(newsletters: list[dict]) -> list[dict]:
    expanded = []
    for email in newsletters:
        source_body = email.get("clean_body") or email.get("body", "")
        chunks      = _chunk_body(source_body)
        if not chunks:
            expanded.append(email)
            continue
        for chunk in chunks:
            expanded.append({**email, "body": chunk, "clean_body": chunk})
    return expanded


def _group_by_topic(newsletters: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for email in newsletters:
        for topic in email.get("topics") or ["general_tech"]:
            groups.setdefault(topic, []).append(email)
    return groups


def _story_text(email: dict) -> str:
    subject = email.get("subject", "")
    body    = email.get("clean_body") or email.get("body", "")
    body    = re.sub(r"\s+", " ", body).strip()
    return f"{subject}. {body[:600]}"


def _embed(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def _embed_all_parallel(texts: list[str]) -> list[list[float]]:
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {pool.submit(_embed, text): i for i, text in enumerate(texts)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _centroid(embeddings: list[list[float]]) -> list[float]:
    n   = len(embeddings)
    dim = len(embeddings[0])
    return [sum(e[i] for e in embeddings) / n for i in range(dim)]


def _sender_name(sender: str) -> str:
    if "<" in sender:
        name = sender.split("<")[0].strip().strip('"')
        return name if name else sender
    return sender.split("@")[0]


# ── Stage 1 — semantic deduplication ─────────────────────────────────────────

def _deduplicate(
    newsletters: list[dict],
    embeddings: list[list[float]],
) -> tuple[list[dict], list[list[float]]]:
    kept_idx: list[int] = []
    for i, emb_i in enumerate(embeddings):
        duplicate = any(
            _cosine(emb_i, embeddings[j]) >= DEDUP_THRESHOLD
            for j in kept_idx
        )
        if not duplicate:
            kept_idx.append(i)
    return [newsletters[i] for i in kept_idx], [embeddings[i] for i in kept_idx]


# ── Stage 2 — greedy story clustering ────────────────────────────────────────

def _cluster_stories(
    newsletters: list[dict],
    embeddings: list[list[float]],
) -> list[list[dict]]:
    clusters: list[list[int]]   = []
    centroids: list[list[float]] = []

    for i, emb in enumerate(embeddings):
        best_cluster, best_score = -1, -1.0
        for ci, centroid in enumerate(centroids):
            score = _cosine(emb, centroid)
            if score > best_score:
                best_score, best_cluster = score, ci

        if best_score >= CLUSTER_THRESHOLD:
            clusters[best_cluster].append(i)
            centroids[best_cluster] = _centroid([embeddings[k] for k in clusters[best_cluster]])
        else:
            clusters.append([i])
            centroids.append(emb)

    clusters.sort(key=len, reverse=True)
    return [[newsletters[i] for i in cluster] for cluster in clusters[:MAX_CLUSTERS]]


# ── Stage 3 — single model call per topic ────────────────────────────────────

def _call_model(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":   OLLAMA_SUMMARIZE_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0, "top_p": 1, "num_predict": 1400},
        },
        timeout=240,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _synthesize_topic_digest(topic: str, clusters: list[list[dict]]) -> dict:
    cluster_blocks = []
    cluster_meta   = []

    for idx, cluster in enumerate(clusters, 1):
        sources    = list({_sender_name(e.get("sender", "")) for e in cluster})
        headline   = cluster[0].get("subject", "")
        n          = len(cluster)
        importance = "high" if n >= 3 else "medium" if n >= 2 else "low"

        excerpts = []
        for e in cluster[:5]:          # cap at 5 representative emails per cluster
            body = e.get("clean_body") or e.get("body", "")
            body = re.sub(r"\s+", " ", body).strip()[:250]
            excerpts.append(f"  - {e.get('subject', '')}: {body}")

        cluster_blocks.append(
            f"STORY {idx} [{importance}] (covered by {n} source(s)):\n" +
            "\n".join(excerpts)
        )
        cluster_meta.append({
            "headline":   headline,
            "sources":    sources,
            "importance": importance,
        })

    stories_text = "\n\n".join(cluster_blocks)

    topic_label = topic.replace("_", " ").title()
    prompt = f"""You are writing a daily news digest for a tech-savvy reader.
Topic: {topic_label}

Stories are grouped by similarity below. For EACH story write a 4-5 sentence summary that explains what happened, names the key people or companies involved, states why it matters, and notes what to watch next.
Then write a 2-3 sentence OVERVIEW of the main theme across all stories.

{stories_text}

Reply in this EXACT format (no preamble, no extra text):
OVERVIEW: <2-3 sentence theme and significance>
STORY 1: <4-5 sentence summary>
STORY 2: <4-5 sentence summary>
...continue for every story above.
"""

    raw = _call_model(prompt)

    overview_match = re.search(r"OVERVIEW:\s*(.+?)(?=STORY\s*1:|$)", raw, re.DOTALL | re.IGNORECASE)
    overview = overview_match.group(1).strip() if overview_match else ""

    sections = []
    for idx, meta in enumerate(cluster_meta, 1):
        story_match = re.search(
            rf"STORY\s*{idx}:\s*(.+?)(?=STORY\s*{idx+1}:|$)", raw, re.DOTALL | re.IGNORECASE
        )
        summary = story_match.group(1).strip() if story_match else ""
        sections.append({
            "headline":   meta["headline"],
            "summary":    summary,
            "importance": meta["importance"],
            "sources":    meta["sources"],
        })

    return {"overview": overview, "sections": sections}


# ── incremental merge helpers ─────────────────────────────────────────────────

def _merge_into_topic(topic: str, new_clusters: list[list[dict]], existing: dict) -> dict:
    """Synthesize new clusters and append them into an already-existing topic."""
    new_data = _synthesize_topic_digest(topic, new_clusters)

    old_overview = existing.get("overview", "")
    new_overview = new_data.get("overview", "")
    if old_overview and new_overview:
        merged_overview = f"{old_overview} {new_overview}"
    else:
        merged_overview = old_overview or new_overview

    return {
        "overview": merged_overview,
        "sections": existing.get("sections", []) + new_data.get("sections", []),
    }


# ── embedding persistence ─────────────────────────────────────────────────────

def _store_embeddings(newsletters: list[dict], embeddings: list[list[float]]):
    """Persist chunk embeddings to the DB for later RAG retrieval."""
    for email, emb in zip(newsletters, embeddings):
        email_id = email.get("id")
        if not email_id:
            continue
        chunk_idx = email.get("_chunk_index", 0)
        chunk_text = _story_text(email)
        db.save_embedding(email_id, chunk_idx, chunk_text, emb)


# ── main orchestrator ─────────────────────────────────────────────────────────

def generate_daily_digest(user_id: int, gmail_account_id: int = None) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n  Summarizer  |  {today}")
    print("  " + "-" * 56)

    all_newsletters = _load_todays_newsletters(user_id, gmail_account_id=gmail_account_id)
    if not all_newsletters:
        print("  No newsletters found for today")
        return {}

    # Load existing digest — ignore it if synthesis previously failed (no topics saved)
    existing_digest = db.get_todays_digest(user_id, gmail_account_id=gmail_account_id)
    if existing_digest and not existing_digest.get("topics"):
        print("  Previous digest has no topics (synthesis failed) — will retry all emails")
        existing_digest = None
    processed_ids: set = set()
    if existing_digest:
        processed_ids = db.get_processed_email_ids(user_id, gmail_account_id=gmail_account_id)
        print(f"  Existing digest — {len(processed_ids)} email(s) already processed")

    # Only process emails not yet in the digest
    new_newsletters = [e for e in all_newsletters if e.get("id") not in processed_ids]
    if not new_newsletters:
        print("  No new emails since last digest — returning cached digest")
        return existing_digest or {}

    print(f"  {len(all_newsletters)} total  |  {len(new_newsletters)} new to process")

    new_newsletters = _expand_newsletters(new_newsletters)
    chunk_counts: dict = {}
    for e in new_newsletters:
        eid = e.get("id")
        if eid:
            chunk_counts[eid] = chunk_counts.get(eid, 0)
            e["_chunk_index"] = chunk_counts[eid]
            chunk_counts[eid] += 1

    print(f"  {len(new_newsletters)} chunk(s) after expansion")

    # Load pre-computed embeddings from DB; only call the model for missing chunks
    embed_map: dict = {}
    to_compute: list[int] = []

    stored_rows = db.get_embeddings_for_user(user_id, gmail_account_id=gmail_account_id)
    stored_lookup = {(r["email_id"], r["chunk_index"]): r["embedding"] for r in stored_rows}
    for i, e in enumerate(new_newsletters):
        key = (e.get("id"), e.get("_chunk_index", 0))
        if key in stored_lookup:
            embed_map[key] = stored_lookup[key]
        else:
            to_compute.append(i)

    if to_compute:
        print(f"  Embedding {len(to_compute)} chunk(s) via embed model...", end="", flush=True)
        try:
            texts    = [_story_text(new_newsletters[i]) for i in to_compute]
            computed = _embed_all_parallel(texts)
            for i, emb in zip(to_compute, computed):
                e   = new_newsletters[i]
                key = (e.get("id"), e.get("_chunk_index", 0))
                embed_map[key] = emb
            _store_embeddings([new_newsletters[i] for i in to_compute], computed)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            return existing_digest or {}
        print("  done")
    else:
        print(f"  All {len(new_newsletters)} embeddings loaded from DB — skipping embed model")

    topic_groups = _group_by_topic(new_newsletters)
    print(f"  {len(topic_groups)} topic(s) in new emails: {', '.join(topic_groups)}")
    print("  " + "-" * 56)

    # Start from existing topics so we can merge into them
    digest_topics: dict[str, dict] = dict(existing_digest.get("topics", {})) if existing_digest else {}

    for topic, emails in topic_groups.items():
        print(f"\n  [{topic}]  {len(emails)} new email(s)")
        embeddings = [embed_map[(e.get("id"), e.get("_chunk_index", 0))] for e in emails]
        emails, embeddings = _deduplicate(emails, embeddings)
        print(f"    {len(emails)} unique after deduplication")
        if not emails:
            continue

        clusters = _cluster_stories(emails, embeddings)[:MAX_CLUSTERS]
        print(f"    {len(clusters)} cluster(s)  (sizes: {[len(c) for c in clusters]})")
        print(f"    synthesizing...", end="", flush=True)
        try:
            if topic in digest_topics:
                # Topic already has a digest — merge new content in
                digest_topics[topic] = _merge_into_topic(topic, clusters, digest_topics[topic])
            else:
                # Brand-new topic — generate from scratch
                digest_topics[topic] = _synthesize_topic_digest(topic, clusters)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            continue
        print("  done")

    # If every topic synthesis failed, don't save — leave emails unprocessed so user can retry
    if not digest_topics:
        print("  All synthesis attempts failed — digest NOT saved.")
        print("  Emails remain unprocessed. Fix Ollama and click Generate Digest again.")
        return existing_digest or {}

    digest = {
        "date":         today,
        "generated_at": datetime.now().isoformat(),
        "topics":       digest_topics,
    }

    new_email_ids = list({e.get("id") for e in new_newsletters if e.get("id")})
    digest_id = db.save_digest(user_id, digest, gmail_account_id=gmail_account_id)
    db.mark_emails_processed(digest_id, new_email_ids)
    print(f"  Digest saved -> PostgreSQL (user_id={user_id})")

    print(f"\n  " + "-" * 56)
    print(f"  Topics: {', '.join(digest_topics) or 'none'}")
    print(f"  New emails processed: {len(new_email_ids)}\n")

    return digest


if __name__ == "__main__":
    # Resolve first user in DB as default
    user_id = None
    try:
        with db.get_cursor() as cur:
            cur.execute("SELECT id FROM users LIMIT 1")
            row = cur.fetchone()
            if row:
                user_id = row["id"]
    except Exception:
        pass

    generate_daily_digest(user_id)
