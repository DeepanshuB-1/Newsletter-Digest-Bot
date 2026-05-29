"""
db.py — PostgreSQL database layer for Newsletter Bot.

Handles:
  - Connection management
  - Schema creation (all 10 tables)
  - CRUD for users, emails, digests, embeddings
"""

import os
import json
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from dotenv import load_dotenv

try:
    from pgvector.psycopg2 import register_vector as _register_vector
    _PGVECTOR = True
except ImportError:
    _PGVECTOR = False

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "newsletter_bot"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "1234"),
}


# ── connection ────────────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    if _PGVECTOR:
        try:
            _register_vector(conn)  # no-op if vector extension not in PG
        except Exception:
            pass
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur


# ── schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(255) NOT NULL,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login    TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gmail_accounts (
    id               SERIAL PRIMARY KEY,
    user_id          INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    gmail_address    VARCHAR(255) NOT NULL,
    app_password_enc TEXT,
    refresh_token    TEXT,
    is_active        BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS emails (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    sender        VARCHAR(500),
    subject       TEXT,
    date_received VARCHAR(255),
    body          TEXT,
    html_body     TEXT,
    clean_body    TEXT,
    has_html      BOOLEAN DEFAULT FALSE,
    body_hash     VARCHAR(64),
    category      VARCHAR(50),
    fetched_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, body_hash)
);

CREATE TABLE IF NOT EXISTS email_topics (
    id       SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    topic    VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS email_images (
    id       SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    url      TEXT NOT NULL,
    position INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS email_embeddings (
    id          SERIAL PRIMARY KEY,
    email_id    INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    chunk_index INTEGER DEFAULT 0,
    chunk_text  TEXT,
    embedding   JSONB
);

CREATE TABLE IF NOT EXISTS digests (
    id           SERIAL PRIMARY KEY,
    user_id      INTEGER REFERENCES users(id) ON DELETE CASCADE,
    date         DATE NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date)
);

CREATE TABLE IF NOT EXISTS digest_topics (
    id         SERIAL PRIMARY KEY,
    digest_id  INTEGER REFERENCES digests(id) ON DELETE CASCADE,
    topic_name VARCHAR(100) NOT NULL,
    overview   TEXT
);

CREATE TABLE IF NOT EXISTS digest_sections (
    id              SERIAL PRIMARY KEY,
    digest_topic_id INTEGER REFERENCES digest_topics(id) ON DELETE CASCADE,
    headline        TEXT,
    summary         TEXT,
    importance      VARCHAR(20) DEFAULT 'medium',
    position        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS digest_section_sources (
    id          SERIAL PRIMARY KEY,
    section_id  INTEGER REFERENCES digest_sections(id) ON DELETE CASCADE,
    source_name VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_emails_user_id       ON emails(user_id);
CREATE INDEX IF NOT EXISTS idx_emails_category      ON emails(category);
CREATE INDEX IF NOT EXISTS idx_emails_fetched_at    ON emails(fetched_at);
CREATE INDEX IF NOT EXISTS idx_email_topics_email   ON email_topics(email_id);
CREATE INDEX IF NOT EXISTS idx_email_topics_topic   ON email_topics(topic);
CREATE INDEX IF NOT EXISTS idx_digests_user_date    ON digests(user_id, date);
CREATE INDEX IF NOT EXISTS idx_embeddings_email     ON email_embeddings(email_id);

CREATE TABLE IF NOT EXISTS digest_emails (
    id         SERIAL PRIMARY KEY,
    digest_id  INTEGER REFERENCES digests(id) ON DELETE CASCADE,
    email_id   INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    UNIQUE(digest_id, email_id)
);

CREATE INDEX IF NOT EXISTS idx_digest_emails_digest ON digest_emails(digest_id);

CREATE TABLE IF NOT EXISTS email_links (
    id          SERIAL PRIMARY KEY,
    email_id    INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    url         TEXT NOT NULL,
    anchor_text TEXT,
    position    INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_email_links_email ON email_links(email_id);

CREATE TABLE IF NOT EXISTS sender_classifications (
    id           SERIAL PRIMARY KEY,
    user_id      INTEGER REFERENCES users(id) ON DELETE CASCADE,
    sender_email VARCHAR(255) NOT NULL,
    category     VARCHAR(50) NOT NULL,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, sender_email)
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id               SERIAL PRIMARY KEY,
    user_id          INTEGER REFERENCES users(id) ON DELETE CASCADE,
    gmail_account_id INTEGER REFERENCES gmail_accounts(id) ON DELETE CASCADE,
    title            VARCHAR(255) NOT NULL DEFAULT 'New Chat',
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       VARCHAR(20) NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);
"""


def create_schema():
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Enable pgvector extension if available on this PostgreSQL instance
            cur.execute("""
                DO $$ BEGIN
                    CREATE EXTENSION IF NOT EXISTS vector;
                EXCEPTION WHEN OTHERS THEN
                    RAISE NOTICE 'pgvector extension not available — vector search disabled';
                END $$;
            """)
            cur.execute(SCHEMA_SQL)
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'email_embeddings_email_chunk_uq'
                    ) THEN
                        ALTER TABLE email_embeddings
                        ADD CONSTRAINT email_embeddings_email_chunk_uq
                        UNIQUE (email_id, chunk_index);
                    END IF;
                END $$;
            """)
            # Add embedding_vec vector column if pgvector extension is present
            cur.execute("""
                DO $$ BEGIN
                    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')
                    AND NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'email_embeddings' AND column_name = 'embedding_vec'
                    ) THEN
                        ALTER TABLE email_embeddings ADD COLUMN embedding_vec vector(768);
                        -- Backfill from JSONB for any existing rows
                        UPDATE email_embeddings
                           SET embedding_vec = (embedding::text)::vector
                         WHERE embedding IS NOT NULL;
                    END IF;
                END $$;
            """)
            # hnsw index for cosine ANN search (no minimum row count unlike ivfflat)
            cur.execute("""
                DO $$ BEGIN
                    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')
                    AND NOT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE tablename = 'email_embeddings'
                          AND indexname = 'idx_embeddings_vec_hnsw'
                    ) THEN
                        EXECUTE 'CREATE INDEX idx_embeddings_vec_hnsw
                                 ON email_embeddings
                                 USING hnsw (embedding_vec vector_cosine_ops)';
                    END IF;
                END $$;
            """)
            # Migrate sender_classifications to per-user if not already done
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'sender_classifications'
                          AND column_name = 'user_id'
                    ) THEN
                        DELETE FROM sender_classifications;
                        ALTER TABLE sender_classifications
                            ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;
                        ALTER TABLE sender_classifications
                            DROP CONSTRAINT IF EXISTS sender_classifications_sender_email_key;
                        ALTER TABLE sender_classifications
                            ADD CONSTRAINT sender_classifications_user_sender_uq
                            UNIQUE (user_id, sender_email);
                    END IF;
                END $$;
            """)
            # ── gmail_accounts: allow multiple per user (one per gmail address) ──
            cur.execute("""
                DO $$ BEGIN
                    -- Drop old single-account constraint (either naming convention)
                    ALTER TABLE gmail_accounts DROP CONSTRAINT IF EXISTS gmail_accounts_user_id_key;
                    ALTER TABLE gmail_accounts DROP CONSTRAINT IF EXISTS gmail_accounts_user_id_unique;
                    -- Add per-(user, address) uniqueness
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'gmail_accounts_user_gmail_uq'
                    ) THEN
                        ALTER TABLE gmail_accounts
                            ADD CONSTRAINT gmail_accounts_user_gmail_uq
                            UNIQUE (user_id, gmail_address);
                    END IF;
                END $$;
            """)
            # ── emails: add gmail_account_id so emails are scoped per inbox ──────
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'emails' AND column_name = 'gmail_account_id'
                    ) THEN
                        ALTER TABLE emails
                            ADD COLUMN gmail_account_id INTEGER REFERENCES gmail_accounts(id);
                        -- Backfill from gmail_accounts (one-per-user constraint still held)
                        UPDATE emails e
                           SET gmail_account_id = g.id
                          FROM gmail_accounts g
                         WHERE g.user_id = e.user_id;
                        -- Replace old unique with per-account unique
                        ALTER TABLE emails
                            DROP CONSTRAINT IF EXISTS emails_user_id_body_hash_key;
                        ALTER TABLE emails
                            ADD CONSTRAINT emails_gmail_account_body_hash_uq
                            UNIQUE (gmail_account_id, body_hash);
                    END IF;
                END $$;
            """)
            # ── digests: add gmail_account_id so digests are scoped per inbox ───
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'digests' AND column_name = 'gmail_account_id'
                    ) THEN
                        ALTER TABLE digests
                            ADD COLUMN gmail_account_id INTEGER REFERENCES gmail_accounts(id);
                        UPDATE digests d
                           SET gmail_account_id = g.id
                          FROM gmail_accounts g
                         WHERE g.user_id = d.user_id;
                        ALTER TABLE digests
                            DROP CONSTRAINT IF EXISTS digests_user_id_date_key;
                        ALTER TABLE digests
                            ADD CONSTRAINT digests_gmail_account_date_uq
                            UNIQUE (gmail_account_id, date);
                    END IF;
                END $$;
            """)
            # ── chat_sessions: add gmail_account_id for per-inbox scoping ────────
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'chat_sessions' AND column_name = 'gmail_account_id'
                    ) THEN
                        ALTER TABLE chat_sessions
                            ADD COLUMN gmail_account_id INTEGER REFERENCES gmail_accounts(id) ON DELETE CASCADE;
                    END IF;
                END $$;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_gmail
                    ON chat_sessions(gmail_account_id, updated_at DESC);
            """)
    print("  Schema created successfully.")


# ── users ─────────────────────────────────────────────────────────────────────

def create_user(name: str, email: str, password_hash: str) -> dict:
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO users (name, email, password_hash)
               VALUES (%s, %s, %s) RETURNING *""",
            (name, email, password_hash),
        )
        return dict(cur.fetchone())


def get_user_by_email(email: str) -> dict | None:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def update_last_login(user_id: int):
    with get_cursor() as cur:
        cur.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
            (user_id,),
        )


# ── gmail accounts ────────────────────────────────────────────────────────────

def save_gmail_account(user_id: int, gmail_address: str,
                       app_password_enc: str = None, refresh_token: str = None) -> dict:
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO gmail_accounts (user_id, gmail_address, app_password_enc, refresh_token)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (user_id, gmail_address) DO UPDATE
                 SET app_password_enc = EXCLUDED.app_password_enc,
                     refresh_token    = EXCLUDED.refresh_token,
                     is_active        = TRUE
               RETURNING *""",
            (user_id, gmail_address, app_password_enc, refresh_token),
        )
        row = cur.fetchone()
        return dict(row) if row else {}


def get_gmail_account(user_id: int, gmail_address: str = None) -> dict | None:
    """Return the active Gmail account for a user, optionally filtered by address."""
    with get_cursor() as cur:
        if gmail_address:
            cur.execute(
                "SELECT * FROM gmail_accounts WHERE user_id = %s AND gmail_address = %s AND is_active = TRUE",
                (user_id, gmail_address),
            )
        else:
            cur.execute(
                "SELECT * FROM gmail_accounts WHERE user_id = %s AND is_active = TRUE ORDER BY id DESC LIMIT 1",
                (user_id,),
            )
        row = cur.fetchone()
        return dict(row) if row else None


def get_all_gmail_accounts(user_id: int) -> list[dict]:
    """Return all active Gmail accounts connected by this user."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM gmail_accounts WHERE user_id = %s AND is_active = TRUE ORDER BY id",
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ── emails ────────────────────────────────────────────────────────────────────

def save_email(user_id: int, email_data: dict, gmail_account_id: int = None) -> int | None:
    """
    Insert email into DB. Returns the new email id, or None if duplicate.
    gmail_account_id scopes the email to a specific connected inbox.
    """
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO emails
               (user_id, gmail_account_id, sender, subject, date_received, body, html_body,
                clean_body, has_html, body_hash, category, fetched_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (gmail_account_id, body_hash) DO NOTHING
               RETURNING id""",
            (
                user_id,
                gmail_account_id,
                email_data.get("sender"),
                email_data.get("subject"),
                email_data.get("date"),
                email_data.get("body"),
                email_data.get("html_body", ""),
                email_data.get("clean_body", ""),
                email_data.get("has_html", False),
                email_data.get("body_hash"),
                email_data.get("category"),
                email_data.get("fetched_at"),
            ),
        )
        row = cur.fetchone()
        if not row:
            return None

        email_id = row["id"]

        # Insert topics
        for topic in email_data.get("topics", []):
            cur.execute(
                "INSERT INTO email_topics (email_id, topic) VALUES (%s, %s)",
                (email_id, topic),
            )

        # Insert images
        for i, url in enumerate(email_data.get("images", [])):
            cur.execute(
                "INSERT INTO email_images (email_id, url, position) VALUES (%s, %s, %s)",
                (email_id, url, i),
            )

        return email_id


def get_todays_emails(user_id: int, gmail_account_id: int = None, category: str = None) -> list[dict]:
    account_clause = "AND e.gmail_account_id = %s" if gmail_account_id else ""
    with get_cursor() as cur:
        if category:
            params = [user_id] + ([gmail_account_id] if gmail_account_id else []) + [category]
            cur.execute(
                f"""SELECT e.*,
                          ARRAY_AGG(DISTINCT et.topic) FILTER (WHERE et.topic IS NOT NULL) AS topics,
                          ARRAY_AGG(DISTINCT ei.url)   FILTER (WHERE ei.url   IS NOT NULL) AS images
                   FROM emails e
                   LEFT JOIN email_topics et ON et.email_id = e.id
                   LEFT JOIN email_images ei ON ei.email_id = e.id
                   WHERE e.user_id = %s
                     {account_clause}
                     AND e.category = %s
                     AND DATE(e.fetched_at) = CURRENT_DATE
                   GROUP BY e.id
                   ORDER BY e.fetched_at""",
                params,
            )
        else:
            params = [user_id] + ([gmail_account_id] if gmail_account_id else [])
            cur.execute(
                f"""SELECT e.*,
                          ARRAY_AGG(DISTINCT et.topic) FILTER (WHERE et.topic IS NOT NULL) AS topics,
                          ARRAY_AGG(DISTINCT ei.url)   FILTER (WHERE ei.url   IS NOT NULL) AS images
                   FROM emails e
                   LEFT JOIN email_topics et ON et.email_id = e.id
                   LEFT JOIN email_images ei ON ei.email_id = e.id
                   WHERE e.user_id = %s
                     {account_clause}
                     AND DATE(e.fetched_at) = CURRENT_DATE
                   GROUP BY e.id
                   ORDER BY e.fetched_at""",
                params,
            )
        return [dict(r) for r in cur.fetchall()]


def email_exists(gmail_account_id: int, body_hash: str) -> bool:
    with get_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM emails WHERE gmail_account_id = %s AND body_hash = %s",
            (gmail_account_id, body_hash),
        )
        return cur.fetchone() is not None


# ── embeddings ────────────────────────────────────────────────────────────────

def save_embedding(email_id: int, chunk_index: int,
                   chunk_text: str, embedding: list[float]):
    if _PGVECTOR:
        import numpy as np
        try:
            with get_cursor() as cur:
                cur.execute(
                    """INSERT INTO email_embeddings
                           (email_id, chunk_index, chunk_text, embedding, embedding_vec)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (email_id, chunk_index) DO NOTHING""",
                    (email_id, chunk_index, chunk_text,
                     json.dumps(embedding), np.array(embedding)),
                )
            return
        except Exception:
            pass  # embedding_vec column not yet present — fall through to JSONB-only
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO email_embeddings (email_id, chunk_index, chunk_text, embedding)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (email_id, chunk_index) DO NOTHING""",
            (email_id, chunk_index, chunk_text, json.dumps(embedding)),
        )


def get_embeddings_for_user(user_id: int, gmail_account_id: int = None, date_only: bool = True) -> list[dict]:
    """Fetch all chunk embeddings for a user's emails (today by default)."""
    date_filter    = "AND DATE(e.fetched_at) = CURRENT_DATE" if date_only else ""
    account_filter = "AND e.gmail_account_id = %s" if gmail_account_id else ""
    with get_cursor() as cur:
        params = [user_id] + ([gmail_account_id] if gmail_account_id else [])
        cur.execute(
            f"""SELECT ee.id, ee.email_id, ee.chunk_index, ee.chunk_text,
                       ee.embedding, e.subject, e.sender, e.category,
                       ARRAY_AGG(DISTINCT et.topic) FILTER (WHERE et.topic IS NOT NULL) AS topics,
                       ARRAY_AGG(DISTINCT ei.url)   FILTER (WHERE ei.url   IS NOT NULL) AS images
                FROM email_embeddings ee
                JOIN emails e ON e.id = ee.email_id
                LEFT JOIN email_topics et ON et.email_id = e.id
                LEFT JOIN email_images ei ON ei.email_id = e.id
                WHERE e.user_id = %s
                  {account_filter}
                  {date_filter}
                GROUP BY ee.id, e.id""",
            params,
        )
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("embedding"), str):
                d["embedding"] = json.loads(d["embedding"])
            result.append(d)
        return result


def search_similar_chunks(
    user_id: int,
    query_vec: list[float],
    top_k: int = 12,
    date_only: bool = True,
    gmail_account_id: int = None,
) -> list[dict]:
    """
    ANN cosine search via pgvector. Returns up to top_k raw-email chunk dicts
    (same shape as corpus chunks in bot.py). Falls back to empty list if
    pgvector is not installed or no embedding_vec column exists.
    """
    if not _PGVECTOR:
        return []
    import numpy as np

    date_clause    = "AND DATE(e.fetched_at) = CURRENT_DATE" if date_only else ""
    account_clause = "AND e.gmail_account_id = %s" if gmail_account_id else ""
    with get_cursor() as cur:
        try:
            params = [user_id] + ([gmail_account_id] if gmail_account_id else []) + [np.array(query_vec), top_k]
            cur.execute(
                f"""SELECT ee.chunk_text,
                           e.id    AS email_id,
                           e.subject,
                           e.sender,
                           ARRAY_AGG(DISTINCT et.topic)
                             FILTER (WHERE et.topic IS NOT NULL) AS topics,
                           ARRAY_AGG(el.url       ORDER BY el.position)
                             FILTER (WHERE el.url IS NOT NULL)   AS link_urls,
                           ARRAY_AGG(el.anchor_text ORDER BY el.position)
                             FILTER (WHERE el.url IS NOT NULL)   AS link_texts
                    FROM email_embeddings ee
                    JOIN emails e ON e.id = ee.email_id
                    LEFT JOIN email_topics et ON et.email_id = e.id
                    LEFT JOIN email_links  el ON el.email_id = e.id
                    WHERE e.user_id    = %s
                      AND e.category  = 'newsletter'
                      AND ee.embedding_vec IS NOT NULL
                      {account_clause}
                      {date_clause}
                    GROUP BY ee.id, e.id
                    ORDER BY ee.embedding_vec <=> %s
                    LIMIT %s""",
                params,
            )
        except Exception:
            return []

        result = []
        for r in cur.fetchall():
            d = dict(r)
            topics = [t for t in (d.get("topics") or []) if t]
            links  = [
                {"url": u, "text": (t or "")[:60]}
                for u, t in zip(d.get("link_urls") or [], d.get("link_texts") or [])
                if u
            ]
            raw_sender  = d.get("sender", "")
            sender_name = (
                raw_sender.split("<")[0].strip().strip('"')
                if "<" in raw_sender else raw_sender.split("@")[0]
            )
            sender_email = (
                raw_sender.split("<")[1].rstrip(">").strip().lower()
                if "<" in raw_sender else raw_sender.strip().lower()
            )
            result.append({
                "text":         d.get("chunk_text", ""),
                "topic":        topics[0] if topics else "general_tech",
                "headline":     d.get("subject", ""),
                "source":       sender_name,
                "source_email": sender_email,
                "type":         "raw_email",
                "importance":   "low",
                "links":        links,
            })
        return result


# ── digests ───────────────────────────────────────────────────────────────────

def save_digest(user_id: int, digest: dict, gmail_account_id: int = None) -> int:
    """Save full digest structure. Returns digest id."""
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO digests (user_id, gmail_account_id, date, generated_at)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (gmail_account_id, date)
               DO UPDATE SET generated_at = EXCLUDED.generated_at
               RETURNING id""",
            (user_id, gmail_account_id, digest["date"], digest["generated_at"]),
        )
        digest_id = cur.fetchone()["id"]

        # Remove old topics/sections for this digest
        cur.execute("DELETE FROM digest_topics WHERE digest_id = %s", (digest_id,))

        for topic_name, data in digest.get("topics", {}).items():
            cur.execute(
                """INSERT INTO digest_topics (digest_id, topic_name, overview)
                   VALUES (%s, %s, %s) RETURNING id""",
                (digest_id, topic_name, data.get("overview", "")),
            )
            topic_id = cur.fetchone()["id"]

            for pos, section in enumerate(data.get("sections", [])):
                cur.execute(
                    """INSERT INTO digest_sections
                       (digest_topic_id, headline, summary, importance, position)
                       VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                    (topic_id, section["headline"], section["summary"],
                     section.get("importance", "medium"), pos),
                )
                section_id = cur.fetchone()["id"]

                for src in section.get("sources", []):
                    cur.execute(
                        "INSERT INTO digest_section_sources (section_id, source_name) VALUES (%s, %s)",
                        (section_id, src),
                    )

        return digest_id


def get_todays_digest(user_id: int, gmail_account_id: int = None) -> dict | None:
    """Reconstruct full digest dict for a user for today, scoped to a Gmail account if provided."""
    with get_cursor() as cur:
        if gmail_account_id:
            cur.execute(
                "SELECT * FROM digests WHERE gmail_account_id = %s AND date = CURRENT_DATE",
                (gmail_account_id,),
            )
        else:
            cur.execute(
                "SELECT * FROM digests WHERE user_id = %s AND date = CURRENT_DATE ORDER BY id DESC LIMIT 1",
                (user_id,),
            )
        digest_row = cur.fetchone()
        if not digest_row:
            return None

        digest_id = digest_row["id"]
        cur.execute(
            "SELECT * FROM digest_topics WHERE digest_id = %s", (digest_id,)
        )
        topics_rows = cur.fetchall()

        topics = {}
        for t in topics_rows:
            cur.execute(
                "SELECT * FROM digest_sections WHERE digest_topic_id = %s ORDER BY position",
                (t["id"],),
            )
            sections = []
            for s in cur.fetchall():
                cur.execute(
                    "SELECT source_name FROM digest_section_sources WHERE section_id = %s",
                    (s["id"],),
                )
                sources = [r["source_name"] for r in cur.fetchall()]
                sections.append({
                    "headline":   s["headline"],
                    "summary":    s["summary"],
                    "importance": s["importance"],
                    "sources":    sources,
                })
            topics[t["topic_name"]] = {
                "overview": t["overview"],
                "sections": sections,
            }

        return {
            "date":         str(digest_row["date"]),
            "generated_at": str(digest_row["generated_at"]),
            "topics":       topics,
        }


def get_processed_email_ids(user_id: int, gmail_account_id: int = None) -> set:
    """Return email IDs already processed into today's digest for this inbox."""
    with get_cursor() as cur:
        if gmail_account_id:
            cur.execute(
                """SELECT de.email_id
                   FROM digest_emails de
                   JOIN digests d ON d.id = de.digest_id
                   WHERE d.gmail_account_id = %s AND d.date = CURRENT_DATE""",
                (gmail_account_id,),
            )
        else:
            cur.execute(
                """SELECT de.email_id
                   FROM digest_emails de
                   JOIN digests d ON d.id = de.digest_id
                   WHERE d.user_id = %s AND d.date = CURRENT_DATE""",
                (user_id,),
            )
        return {row["email_id"] for row in cur.fetchall()}


def mark_emails_processed(digest_id: int, email_ids: list):
    """Link email IDs to a digest to mark them as already processed."""
    if not email_ids:
        return
    with get_cursor() as cur:
        for eid in email_ids:
            cur.execute(
                "INSERT INTO digest_emails (digest_id, email_id) VALUES (%s, %s) ON CONFLICT (digest_id, email_id) DO NOTHING",
                (digest_id, eid),
            )


# ── sender classification cache ───────────────────────────────────────────────

def get_sender_category(sender_email: str, user_id: int) -> str | None:
    """Return this user's cached category for a sender, or None if unseen."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT category FROM sender_classifications WHERE sender_email = %s AND user_id = %s",
            (sender_email, user_id),
        )
        row = cur.fetchone()
        return row["category"] if row else None


def save_sender_category(sender_email: str, category: str, user_id: int):
    """Upsert the category for a sender address scoped to this user."""
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO sender_classifications (sender_email, category, user_id, updated_at)
               VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
               ON CONFLICT (user_id, sender_email) DO UPDATE
                 SET category   = EXCLUDED.category,
                     updated_at = CURRENT_TIMESTAMP""",
            (sender_email, category, user_id),
        )


# ── email links ──────────────────────────────────────────────────────────────

def save_email_links(email_id: int, links: list[dict]):
    """Bulk-insert extracted article links for an email (skip on conflict)."""
    if not links:
        return
    with get_cursor() as cur:
        for link in links:
            cur.execute(
                """INSERT INTO email_links (email_id, url, anchor_text, position)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                (email_id, link["url"], link.get("anchor_text", ""), link.get("position", 0)),
            )


def get_email_links(email_id: int) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT url, anchor_text FROM email_links WHERE email_id = %s ORDER BY position",
            (email_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_links_for_emails(email_ids: list[int]) -> dict:
    """Bulk-fetch links for a list of email IDs. Returns {email_id: [{url, anchor_text}]}."""
    if not email_ids:
        return {}
    with get_cursor() as cur:
        cur.execute(
            """SELECT email_id, url, anchor_text
               FROM email_links
               WHERE email_id = ANY(%s)
               ORDER BY email_id, position""",
            (email_ids,),
        )
        result: dict = {}
        for row in cur.fetchall():
            d = dict(row)
            eid = d.pop("email_id")
            result.setdefault(eid, []).append(d)
    return result


# ── chat sessions ─────────────────────────────────────────────────────────────

def create_chat_session(user_id: int, title: str = "New Chat", gmail_account_id: int = None) -> int:
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO chat_sessions (user_id, gmail_account_id, title) VALUES (%s, %s, %s) RETURNING id",
            (user_id, gmail_account_id, title[:200]),
        )
        return cur.fetchone()["id"]


def update_chat_session_title(session_id: int, title: str):
    with get_cursor() as cur:
        cur.execute(
            "UPDATE chat_sessions SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
            (title[:200], session_id),
        )


def append_chat_message(session_id: int, role: str, content: str):
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, role, content),
        )
        cur.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
            (session_id,),
        )


def get_recent_sessions(user_id: int, gmail_account_id: int = None, limit: int = 10) -> list[dict]:
    with get_cursor() as cur:
        if gmail_account_id:
            cur.execute(
                """SELECT id, title, created_at, updated_at
                   FROM chat_sessions
                   WHERE user_id = %s AND gmail_account_id = %s
                   ORDER BY updated_at DESC
                   LIMIT %s""",
                (user_id, gmail_account_id, limit),
            )
        else:
            cur.execute(
                """SELECT id, title, created_at, updated_at
                   FROM chat_sessions
                   WHERE user_id = %s AND gmail_account_id IS NULL
                   ORDER BY updated_at DESC
                   LIMIT %s""",
                (user_id, limit),
            )
        return [dict(r) for r in cur.fetchall()]


def get_session_messages(session_id: int) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def delete_chat_session(session_id: int):
    with get_cursor() as cur:
        cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))


# ── init ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Creating database schema...")
    create_schema()
    print("  Done.\n")
