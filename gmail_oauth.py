"""
gmail_oauth.py — Google OAuth 2.0 flow + Gmail API email fetching.

Replaces the IMAP/app-password approach with proper per-user OAuth.
Each user connects their own Gmail via the Google consent screen.
Tokens are stored in the gmail_accounts table (refresh_token column as JSON).
"""

import os
import io
import re
import base64
import json
import hashlib
import contextlib
import urllib.parse
from datetime import datetime
from bs4 import BeautifulSoup
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request, AuthorizedSession
from email.header import decode_header as _decode_header
from dotenv import load_dotenv

import db
from classification import classify_email
from news_letter_classifier import classify_newsletter_topic
from cleaner import clean_body

load_dotenv()

SCOPES       = ["https://www.googleapis.com/auth/gmail.readonly"]
REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8501")

_CLIENT_CONFIG = {
    "web": {
        "client_id":     os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
        "token_uri":     "https://oauth2.googleapis.com/token",
        "redirect_uris": [REDIRECT_URI],
    }
}


# ── OAuth flow ────────────────────────────────────────────────────────────────

def get_auth_url(user_id: int) -> str:
    """
    Generate the Google consent-screen URL.
    If google-auth-oauthlib auto-generates a PKCE code verifier, we embed it
    in the state parameter (state = "<user_id>|<verifier>") so it survives
    the redirect — Streamlit loses session state on the OAuth callback.
    """
    flow = Flow.from_client_config(_CLIENT_CONFIG, scopes=SCOPES)
    flow.redirect_uri = REDIRECT_URI
    url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=str(user_id),
        prompt="consent",
    )
    code_verifier = flow.code_verifier  # set by google-auth-oauthlib 1.4+ automatically
    if code_verifier:
        # Rebuild URL with state = "<user_id>|<verifier>" so it survives the redirect
        combined_state = f"{user_id}|{code_verifier}"
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        params["state"] = [combined_state]
        new_query = urllib.parse.urlencode({k: v[0] for k, v in params.items()})
        url = urllib.parse.urlunparse(parsed._replace(query=new_query))
    return url


def exchange_code(code: str, state: str) -> tuple[int, dict]:
    """
    Exchange the auth code returned by Google for tokens.
    State is either "<user_id>" or "<user_id>|<pkce_verifier>".
    Returns (user_id, token_data_dict).
    """
    if "|" in state:
        user_id_str, code_verifier = state.split("|", 1)
    else:
        user_id_str, code_verifier = state, None

    user_id = int(user_id_str)
    flow = Flow.from_client_config(_CLIENT_CONFIG, scopes=SCOPES, state=state)
    flow.redirect_uri = REDIRECT_URI
    if code_verifier:
        flow.code_verifier = code_verifier  # must be set on Flow, not oauth2session
    flow.fetch_token(code=code)
    creds = flow.credentials
    token_data = {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri":     creds.token_uri,
        "client_id":     creds.client_id,
        "client_secret": creds.client_secret,
        "scopes":        list(creds.scopes or SCOPES),
    }
    return user_id, token_data


def _credentials_from_account(account: dict) -> Credentials | None:
    """Build and auto-refresh credentials from a gmail_accounts row dict."""
    if not account or not account.get("refresh_token"):
        return None
    try:
        token_data = json.loads(account["refresh_token"])
    except (json.JSONDecodeError, TypeError):
        return None

    creds = Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_data.get("client_id", os.getenv("GOOGLE_CLIENT_ID")),
        client_secret=token_data.get("client_secret", os.getenv("GOOGLE_CLIENT_SECRET")),
        scopes=token_data.get("scopes", SCOPES),
    )

    if not creds.valid and creds.refresh_token:
        try:
            creds.refresh(Request())
            _update_stored_token(
                account["user_id"], creds,
                gmail_address=account.get("gmail_address"),
            )
        except Exception:
            return None

    return creds if creds.valid else None


def get_credentials(user_id: int) -> Credentials | None:
    """Load credentials for the most recently active Gmail account of a user."""
    return _credentials_from_account(db.get_gmail_account(user_id))


def _update_stored_token(user_id: int, creds: Credentials, gmail_address: str = None):
    updated = {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri":     creds.token_uri,
        "client_id":     creds.client_id,
        "client_secret": creds.client_secret,
        "scopes":        list(creds.scopes or SCOPES),
    }
    with db.get_cursor() as cur:
        if gmail_address:
            cur.execute(
                "UPDATE gmail_accounts SET refresh_token = %s WHERE user_id = %s AND gmail_address = %s",
                (json.dumps(updated), user_id, gmail_address),
            )
        else:
            cur.execute(
                "UPDATE gmail_accounts SET refresh_token = %s WHERE user_id = %s",
                (json.dumps(updated), user_id),
            )


_GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"


def _gmail_session(creds: Credentials) -> AuthorizedSession:
    return AuthorizedSession(creds)


def get_gmail_address(creds: Credentials) -> str:
    session = _gmail_session(creds)
    resp = session.get(f"{_GMAIL_BASE}/profile")
    resp.raise_for_status()
    return resp.json().get("emailAddress", "")


def is_connected(user_id: int) -> bool:
    """True if the user has a valid Gmail connection stored."""
    account = db.get_gmail_account(user_id)
    return bool(account and account.get("refresh_token"))


# ── link extraction ──────────────────────────────────────────────────────────

_SKIP_LINK_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com", "linkedin.com",
    "youtube.com", "t.co", "bit.ly", "ow.ly", "fb.com", "tiktok.com",
}
_SKIP_LINK_RE = re.compile(
    r"unsubscribe|optout|opt-out|manage.{0,15}email|mailto:|"
    r"preferences|view.{0,10}(online|browser)|tracking\.|pixel\.",
    re.IGNORECASE,
)
_GENERIC_ANCHORS = {"click here", "read more", "here", "view online", "learn more",
                    "more", "this", "link", "article", "continue reading"}


def _extract_links(html: str) -> list[dict]:
    """Extract up to 5 meaningful article links from an HTML email."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    links, seen = [], set()
    for pos, tag in enumerate(soup.find_all("a", href=True)):
        url = tag["href"].strip()
        if not url.startswith("http"):
            continue
        try:
            domain = urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
        except Exception:
            continue
        if domain in _SKIP_LINK_DOMAINS:
            continue
        if _SKIP_LINK_RE.search(url):
            continue
        # Deduplicate by domain+path (ignore query strings like utm_*)
        parsed   = urllib.parse.urlparse(url)
        url_key  = f"{parsed.netloc}{parsed.path}".rstrip("/").lower()
        if url_key in seen:
            continue
        anchor = tag.get_text(strip=True)
        if anchor.lower() in _GENERIC_ANCHORS or len(anchor) < 10:
            anchor = ""
        seen.add(url_key)
        links.append({"url": url, "anchor_text": anchor, "position": pos})
        if len(links) >= 5:
            break
    return links


# ── email parsing ─────────────────────────────────────────────────────────────

def _decode_mime_header(raw: str) -> str:
    parts = _decode_header(raw or "")
    out = []
    for part, enc in parts:
        if isinstance(part, bytes):
            out.append(part.decode(enc or "utf-8", errors="ignore"))
        else:
            out.append(part)
    return "".join(out)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
    return "\n".join(lines)


def _body_hash(body: str) -> str:
    normalized = re.sub(r"\s+", " ", body[:600]).strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()


def _extract_parts(payload: dict) -> tuple[str, str]:
    """Recursively pull plain-text and HTML from a Gmail message payload."""
    plain = ""
    html  = ""
    mime  = payload.get("mimeType", "")
    data  = payload.get("body", {}).get("data", "")

    if mime == "text/plain" and data:
        plain = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
    elif mime == "text/html" and data:
        html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")

    for part in payload.get("parts", []):
        p, h = _extract_parts(part)
        if p and not plain:
            plain = p
        if h and not html:
            html = h

    return plain, html


def _parse_gmail_message(msg: dict) -> dict:
    payload = msg.get("payload", {})
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

    subject    = _decode_mime_header(headers.get("subject", "No Subject"))
    sender     = headers.get("from", "Unknown")
    date       = headers.get("date", "")
    plain, html = _extract_parts(payload)

    body = plain if plain.strip() else (_html_to_text(html) if html.strip() else "")

    return {
        "sender":     sender,
        "subject":    subject,
        "date":       date,
        "body":       body,
        "html_body":  html,
        "clean_body": clean_body(body),
        "images":     [],
        "has_html":   bool(html),
        "links":      _extract_links(html),
    }


# ── fetch + classify pipeline ─────────────────────────────────────────────────

@contextlib.contextmanager
def _capture_to_log(log_fn=None):
    """
    Redirect stdout during classification to the progress log instead of
    silently swallowing it. Warnings like 'Cannot connect to Ollama' now
    appear in the sidebar fetch log rather than disappearing completely.
    If no log_fn is provided, output is suppressed (same as the old _quiet).
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    captured = buf.getvalue().strip()
    if captured and log_fn:
        for line in captured.splitlines():
            line = line.strip()
            if line:
                log_fn(f"    {line}")


def run_fetch_pipeline(user_id: int, gmail_account_id: int = None, progress_cb=None) -> dict:
    """
    Fetch today's emails via Gmail API, classify, and store in DB.
    gmail_account_id pins the fetch to a specific connected inbox; if omitted,
    uses the most recently active account for the user.
    progress_cb(msg: str) is called with status updates if provided.
    Returns stats dict: {fetched, saved, skipped, errors}.
    """
    def _log(msg):
        if progress_cb:
            progress_cb(msg)

    # Resolve the specific Gmail account to fetch from
    if gmail_account_id:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM gmail_accounts WHERE id = %s AND is_active = TRUE",
                (gmail_account_id,),
            )
            row = cur.fetchone()
            account = dict(row) if row else None
    else:
        account = db.get_gmail_account(user_id)

    if not account:
        return {"error": "Gmail account not found or inactive. Please reconnect."}

    gmail_account_id = account["id"]
    creds = _credentials_from_account(account)
    if not creds:
        return {"error": "Gmail not connected or token expired. Please reconnect."}

    session = _gmail_session(creds)
    today   = datetime.now().strftime("%Y/%m/%d")

    _log("Fetching email list from Gmail...")
    result = session.get(
        f"{_GMAIL_BASE}/messages",
        params={"q": f"after:{today}", "maxResults": 50},
    ).json()

    messages = result.get("messages", [])
    stats    = {"fetched": len(messages), "saved": 0, "skipped": 0, "errors": 0}

    if not messages:
        _log("No emails found for today.")
        return stats

    _log(f"Found {len(messages)} email(s). Classifying...")

    for i, item in enumerate(messages, 1):
        try:
            msg_data = session.get(
                f"{_GMAIL_BASE}/messages/{item['id']}",
                params={"format": "full"},
            ).json()

            e         = _parse_gmail_message(msg_data)
            body_hash = _body_hash(e.get("body", ""))

            if gmail_account_id and db.email_exists(gmail_account_id, body_hash):
                stats["skipped"] += 1
                _log(f"  [{i}/{len(messages)}] {e['subject'][:45]} — duplicate")
                continue

            with _capture_to_log(_log):
                e["category"] = classify_email(e, user_id)
                if e["category"] == "newsletter":
                    e["topics"] = classify_newsletter_topic(e)
                else:
                    e["topics"] = []

            e["fetched_at"] = datetime.now().isoformat()
            e["body_hash"]  = body_hash

            email_id = db.save_email(user_id, e, gmail_account_id=gmail_account_id)
            if email_id:
                stats["saved"] += 1
                tag = e["category"]
                if e.get("topics"):
                    tag += f" | {', '.join(e['topics'])}"
                _log(f"  [{i}/{len(messages)}] {e['subject'][:45]} — {tag}")

                # Save extracted article links
                if e.get("links"):
                    db.save_email_links(email_id, e["links"])

                # Pre-compute embeddings for newsletters so digest generation skips Ollama
                if e["category"] == "newsletter":
                    try:
                        from summarizer import _chunk_body, _embed, _story_text
                        full_body = e.get("clean_body") or e.get("body", "")
                        chunks    = _chunk_body(full_body) or ([full_body[:700]] if full_body.strip() else [])
                        for idx, chunk in enumerate(chunks):
                            chunk_dict = {**e, "body": chunk, "clean_body": chunk}
                            text = _story_text(chunk_dict)
                            emb  = _embed(text)
                            db.save_embedding(email_id, idx, text, emb)
                        _log(f"    embedded {len(chunks)} chunk(s)")
                    except Exception as emb_ex:
                        _log(f"    embedding skipped: {emb_ex}")
            else:
                stats["skipped"] += 1

        except Exception as ex:
            stats["errors"] += 1
            _log(f"  [{i}/{len(messages)}] ERROR: {ex}")

    return stats
