"""
LEGACY — IMAP / App-Password ingestion path.  NOT used by the running app.

The active Layer 1 implementation is gmail_oauth.py (Google OAuth 2.0).
That path supports per-user authentication, no hardcoded credentials, and
is what bot.py calls via gmail_oauth.run_fetch_pipeline().

This file is kept for reference only.  Do not import it from other modules.
To run the current fetch pipeline use: python -c "import gmail_oauth; ..."
or the 'Fetch Today's Emails' button in the Streamlit UI.
"""
import imaplib
import email
import io
import re
import hashlib
import contextlib
from email.header import decode_header
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from classification import classify_email
from news_letter_classifier import classify_newsletter_topic
from cleaner import clean_body
import db

load_dotenv()
GMAIL_USER     = os.getenv("GMAIL_USER")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
IMAP_SERVER     = "imap.gmail.com"
IMAP_PORT       = 993
JSON_FILE_ALL   = os.getenv("JSON_FILE_ALL",   "emails_all.json")
JSON_FILE_TODAY = os.getenv("JSON_FILE_TODAY", "emails_today.json")

_W = 58


def _div():
    print("  " + "-" * _W)

def _section(title):
    print()
    print("  " + "=" * _W)
    print(f"  {'':2}{title}")
    print("  " + "=" * _W)
    print()

def _field(label, value, indent=4):
    label_col = f"{' ' * indent}{label:<10}"
    print(f"  {label_col}{value}")

@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def decode_subject(raw_subject):
    decoded, encoding = decode_header(raw_subject)[0]
    if isinstance(decoded, bytes):
        return decoded.decode(encoding or "utf-8")
    return decoded


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
    return "\n".join(lines)


def fetch_email_content(mail, msg_id):
    _, msg_data = mail.fetch(msg_id, "(BODY.PEEK[])")
    msg = email.message_from_bytes(msg_data[0][1])

    subject    = decode_subject(msg.get("Subject", "No Subject"))
    sender     = msg.get("From", "Unknown")
    date       = msg.get("Date", "Unknown")
    plain_body = ""
    html_body  = ""

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain" and not plain_body:
                plain_body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
            elif ct == "text/html" and not html_body:
                html_body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            raw = payload.decode("utf-8", errors="ignore")
            if msg.get_content_type() == "text/html":
                html_body = raw
            else:
                plain_body = raw

    if plain_body.strip():
        body = plain_body
    elif html_body.strip():
        body = _html_to_text(html_body)
    else:
        body = ""

    return {
        "sender":     sender,
        "subject":    subject,
        "date":       date,
        "body":       body,
        "html_body":  html_body,
        "clean_body": clean_body(body),
        "images":     [],
        "has_html":   bool(html_body),
    }


def _body_hash(body: str) -> str:
    normalized = re.sub(r"\s+", " ", body[:600]).strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()


# ── JSON helpers (kept for backward compatibility) ────────────────────────────

def _load(path: str) -> list:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _dump(path: str, data: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _reset_today_if_stale():
    today = datetime.now().strftime("%Y-%m-%d")
    entries = _load(JSON_FILE_TODAY)
    if entries:
        first_date = entries[0].get("fetched_at", "")[:10]
        if first_date != today:
            _dump(JSON_FILE_TODAY, [])


def _is_duplicate_json(email_data: dict, store: list) -> bool:
    new_hash = email_data.get("body_hash") or _body_hash(email_data.get("body", ""))
    return any(
        e.get("sender") == email_data["sender"] and
        e.get("subject") == email_data["subject"] and
        (e.get("date") == email_data["date"] or e.get("body_hash") == new_hash)
        for e in store
    )


def _save_to_json(email_data: dict) -> bool:
    all_emails = _load(JSON_FILE_ALL)
    if _is_duplicate_json(email_data, all_emails):
        return False
    all_emails.append(email_data)
    _dump(JSON_FILE_ALL, all_emails)
    today_emails = _load(JSON_FILE_TODAY)
    today_emails.append(email_data)
    _dump(JSON_FILE_TODAY, today_emails)
    return True


# ── DB-backed email save ───────────────────────────────────────────────────────

def _save_email_to_db(user_id: int, e: dict) -> bool:
    """Save email to PostgreSQL. Returns True if newly inserted."""
    body_hash = e.get("body_hash") or _body_hash(e.get("body", ""))
    if db.email_exists(user_id, body_hash):
        return False

    email_data = {
        "sender":     e.get("sender", ""),
        "subject":    e.get("subject", ""),
        "date":       e.get("date", ""),
        "body":       e.get("body", ""),
        "html_body":  e.get("html_body", ""),
        "clean_body": e.get("clean_body", ""),
        "has_html":   e.get("has_html", False),
        "body_hash":  body_hash,
        "category":   e.get("category", "unknown"),
        "fetched_at": e.get("fetched_at", datetime.now().isoformat()),
        "topics":     e.get("topics", []),
        "images":     e.get("images", []),
    }
    return db.save_email(user_id, email_data) is not None


def _get_cached_from_db(user_id: int, e: dict) -> dict | None:
    """Check DB for an existing classified version of this email."""
    body_hash = _body_hash(e.get("body", ""))
    with db.get_cursor() as cur:
        cur.execute(
            """SELECT e.*, ARRAY_AGG(DISTINCT et.topic) FILTER (WHERE et.topic IS NOT NULL) AS topics
               FROM emails e
               LEFT JOIN email_topics et ON et.email_id = e.id
               WHERE e.user_id = %s AND e.body_hash = %s
                 AND e.category IS NOT NULL AND e.category != 'unknown'
               GROUP BY e.id
               LIMIT 1""",
            (user_id, body_hash),
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ── main fetch ────────────────────────────────────────────────────────────────

def fetch_todays_emails(user_id: int = None):
    """
    Fetch today's emails from Gmail, classify them, and save to DB + JSON.
    If user_id is None, attempts to use the first user in the DB.
    """
    # Resolve user_id
    if user_id is None:
        with db.get_cursor() as cur:
            cur.execute("SELECT id FROM users LIMIT 1")
            row = cur.fetchone()
            user_id = row["id"] if row else None

    _reset_today_if_stale()
    today = datetime.now().strftime("%d-%b-%Y")

    print()
    print(f"  Gmail Inbox  ·  {today}")
    _div()

    try:
        print("  Connecting to Gmail...", end="", flush=True)
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(GMAIL_USER, GMAIL_PASSWORD)
        print("  connected")
    except imaplib.IMAP4.error as e:
        print(f"\n  ERROR  Login failed: {e}")
        print("         Check GMAIL_USER and GMAIL_PASSWORD in .env")
        return []
    except Exception as e:
        print(f"\n  ERROR  Connection failed: {e}")
        return []

    try:
        mail.select("INBOX")
        _, msg_ids = mail.search(None, f"SINCE {today}")
        ids = msg_ids[0].split()

        if not ids:
            print("  No emails found for today.")
            return []

        print(f"  Found {len(ids)} email(s)  classifying...")
        _div()

        emails  = []
        skipped = 0

        for i, msg_id in enumerate(ids, 1):
            try:
                e = fetch_email_content(mail, msg_id)

                subject_short = e["subject"][:42].ljust(43)
                print(f"  {str(i).rjust(2)}/{len(ids)}  {subject_short}", end="", flush=True)

                # Check DB for existing classification
                cached = _get_cached_from_db(user_id, e) if user_id else None

                if cached:
                    e["category"]   = cached["category"]
                    e["fetched_at"] = str(cached["fetched_at"])
                    if cached.get("topics"):
                        e["topics"] = [t for t in cached["topics"] if t]

                    tag = e["category"]
                    if e.get("topics"):
                        tag += f"  |  {', '.join(e['topics'])}"

                    # Sync to JSON today file
                    today_entries = _load(JSON_FILE_TODAY)
                    if not _is_duplicate_json(e, today_entries):
                        e.setdefault("fetched_at", datetime.now().isoformat())
                        e.setdefault("body_hash",  _body_hash(e.get("body", "")))
                        today_entries.append(e)
                        _dump(JSON_FILE_TODAY, today_entries)
                        print(f"  {tag}  (cached)")
                        emails.append(e)
                    else:
                        print(f"  {tag}  (already today)")
                        skipped += 1

                else:
                    # New email — run classifiers
                    with _quiet():
                        e["category"] = classify_email(e)
                        if e["category"] == "newsletter":
                            e["topics"] = classify_newsletter_topic(e)

                    tag = e["category"]
                    if e.get("topics"):
                        tag += f"  |  {', '.join(e['topics'])}"

                    e["fetched_at"] = datetime.now().isoformat()
                    e["body_hash"]  = _body_hash(e.get("body", ""))

                    # Save to DB (primary) + JSON (secondary)
                    db_saved   = _save_email_to_db(user_id, e) if user_id else False
                    json_saved = _save_to_json(e)

                    if not db_saved and not json_saved:
                        print(f"  {tag}  (duplicate)")
                        skipped += 1
                    else:
                        print(f"  {tag}")
                        emails.append(e)

            except Exception as ex:
                print(f"  ERROR")
                print(f"         {ex}")

        _div()
        print(f"  {len(emails)} saved  |  {skipped} duplicate(s)")
        return emails

    finally:
        mail.logout()


def reclassify_all():
    if not os.path.exists(JSON_FILE_ALL):
        print("  No emails_all.json found.")
        return

    all_emails = _load(JSON_FILE_ALL)

    _section("RECLASSIFY")
    updated = 0
    for e in all_emails:
        if "category" not in e or e["category"] == "unknown":
            subject_short = e.get("subject", "?")[:42].ljust(43)
            print(f"  {subject_short}", end="", flush=True)
            with _quiet():
                e["category"] = classify_email(e)
                if e["category"] == "newsletter" and "topics" not in e:
                    e["topics"] = classify_newsletter_topic(e)
            topics_str = ", ".join(e["topics"]) if e.get("topics") else ""
            print(f"  {e['category']}" + (f"  |  {topics_str}" if topics_str else ""))
            updated += 1

    _dump(JSON_FILE_ALL, all_emails)

    _div()
    print(f"  {updated} email(s) reclassified.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reclassify":
        reclassify_all()
        sys.exit(0)

    results = fetch_todays_emails()

    if not results:
        sys.exit(0)

    _section("SUMMARY")

    for i, e in enumerate(results, 1):
        cat = e.get("category", "unknown")
        print(f"  [{i}]  {cat}")
        if e.get("topics"):
            _field("Topics", ", ".join(e["topics"]))
        _field("From",    e["sender"][:52])
        _field("Subject", e["subject"][:52])
        preview = e["body"][:160].replace("\n", " ").strip()
        _field("Preview", preview[:52])
        if len(preview) > 52:
            _field("",    preview[52:104])
        print()
        _div()
        print()
