import os
import re
import requests
from email.utils import parseaddr
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CLASSIFY_MODEL = os.getenv("OLLAMA_CLASSIFY_MODEL", "mistral")

VALID_CATEGORIES = ["spam", "newsletter", "personal", "promotional", "work", "notification"]


def _extract_address(sender: str) -> str:
    """Pull bare email address out of 'Name <addr>' or plain 'addr' strings."""
    _, addr = parseaddr(sender)
    return addr.lower().strip()


def _clean_body(body: str, max_chars: int = 600) -> str:
    """
    FIX: Strip HTML tags and URLs before sending to the model.
    Previously HTML emails sent raw URLs as context — model saw noise not content.
    """
    if not body:
        return ""

    # If it looks like HTML, strip tags
    if "<html" in body.lower() or "<body" in body.lower() or "<div" in body.lower():
        soup = BeautifulSoup(body, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        body = soup.get_text(separator=" ")

    # Remove URLs
    body = re.sub(r"https?://\S+", "", body)
    # Collapse whitespace
    body = re.sub(r"\s+", " ", body).strip()

    return body[:max_chars]




def _call_mistral(email_data: dict) -> str:
    """Send the email to Mistral via Ollama and return the predicted category."""

    # FIX: clean body before sending — strip HTML/URLs
    clean = _clean_body(email_data.get("body", ""))

    prompt = f"""You are an email classifier. Classify an email into one of:
spam, newsletter, personal, promotional, work, notification

Definitions:
- spam         : unsolicited junk or phishing the user never signed up for
- newsletter   : subscribed editorial content — news digests, article roundups, weekly updates; primary value is information, NO purchase CTA
- personal     : direct message from an individual person (friend, family, colleague)
- promotional  : main goal is to sell — discount codes, "shop now", sale announcements, product offers
- work         : professional correspondence, invoices, job-related, HR, business meetings
- notification : automated transactional message — OTP, order confirmation, shipping update, account alert

Key rules:
- If the email contains articles/news/information and has NO "buy", "shop", "discount", "offer", or "% off" — it is newsletter, not promotional.
- Google security alerts, OTPs, GitHub notifications = notification
- If sender is a real person's name and email is conversational = personal

Examples:
---
From: newsletter@techdigest.io
Subject: Weekly AI Roundup — Top Stories
Body: Hello readers, here are this week's top AI stories: 1) OpenAI releases... 2) Meta announces...
CATEGORY: newsletter
---
From: deals@amazon.com
Subject: 40% off today only — Flash Sale!
Body: Don't miss out! Shop now and save big. Use code SAVE40 at checkout.
CATEGORY: promotional
---
From: noreply@github.com
Subject: Your pull request was merged
Body: Your pull request #42 has been successfully merged into main.
CATEGORY: notification
---
From: Google <no-reply@accounts.google.com>
Subject: Critical security alert
Body: We received a request to help you sign in. If you didn't make this request, review now.
CATEGORY: notification
---
From: Rahul <rahul@gmail.com>
Subject: hey catch up soon?
Body: Hope you are doing well! Are we still planning that trip to Manali?
CATEGORY: personal
---
Now classify this email. Reply with ONLY one line in this exact format:
CATEGORY: <word>

From: {email_data['sender']}
Subject: {email_data['subject']}
Body: {clean}
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_CLASSIFY_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["response"].strip().lower()

        match = re.search(r"category:\s*([a-z]+)", raw)
        category = match.group(1) if match else raw.split()[0].rstrip(".,;:")

        if category not in VALID_CATEGORIES:
            print(f"[classification] Unexpected category '{category}' — defaulting to notification")
            category = "notification"

    except requests.exceptions.ConnectionError:
        print("[classification] Cannot connect to Ollama. Is it running? Run: ollama serve")
        category = "notification"
    except requests.exceptions.Timeout:
        print("[classification] Ollama request timed out after 60s")
        category = "notification"
    except requests.exceptions.RequestException as e:
        print(f"[classification] Ollama request failed: {e}")
        category = "notification"

    return category


def classify_email(email_data: dict, user_id: int = None) -> str:
    """
    Classify an email. Checks this user's sender_classifications first —
    if the sender is known for this user, returns cached category with no model call.
    On a cache miss, calls the model and saves the result for this user's future fetches.
    """
    import db
    sender_addr = _extract_address(email_data.get("sender", ""))

    if user_id is not None:
        cached = db.get_sender_category(sender_addr, user_id)
        if cached:
            print(f"[classification] Cache hit  {sender_addr} → {cached}")
            return cached

    print(f"[classification] New sender {sender_addr} — calling model")
    category = _call_mistral(email_data)

    if user_id is not None:
        try:
            db.save_sender_category(sender_addr, category, user_id)
        except Exception:
            pass

    return category