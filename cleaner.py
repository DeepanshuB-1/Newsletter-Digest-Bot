"""
cleaner.py — Newsletter body cleaner + smart image extractor.

Adds two fields to every email in the JSON files:
  clean_body : boilerplate-stripped, whitespace-normalised plain text
  images     : list of content-relevant image URLs (icons/logos filtered out)

Image filtering is two-stage:
  Stage 1 — heuristics: URL patterns, alt text, known icon CDNs, size attributes
  Stage 2 — AI filter: qwen2.5:14b scores remaining candidates against the
             newsletter subject and keeps only article/content images

Run standalone to backfill existing JSON:
  python cleaner.py
"""

import os
import re
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

JSON_FILE_ALL   = os.getenv("JSON_FILE_ALL",   "emails_all.json")
JSON_FILE_TODAY = os.getenv("JSON_FILE_TODAY", "emails_today.json")
OLLAMA_BASE_URL           = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_IMAGE_FILTER_MODEL = os.getenv("OLLAMA_IMAGE_FILTER_MODEL", "phi4-mini")

# ── boilerplate phrases ───────────────────────────────────────────────────────

_BOILERPLATE = [
    "unsubscribe", "view in browser", "view this email", "view online",
    "privacy policy", "terms of service", "terms & conditions",
    "all rights reserved", "copyright ©", "copyright (c)",
    "you are receiving this", "you received this because",
    "you're receiving this", "manage preferences", "email preferences",
    "update your preferences", "opt out", "opt-out",
    "if you no longer wish", "no longer want to receive",
    "sent to ", "mailing address", "our mailing",
    "add us to your address book", "safe sender",
    "trouble viewing", "display images", "enable images",
    "click here to", "forward to a friend",
    "powered by", "email was sent", "this newsletter",
]

# ── image heuristic filters ───────────────────────────────────────────────────

# CDNs that serve only UI icons / social media assets — never article content
_ICON_CDNS = [
    "static.licdn.com", "platform.linkedin.com", "media.licdn.com",
    "abs.twimg.com", "pbs.twimg.com/profile",
    "static.xx.fbcdn.net", "scontent.fbcdn.net",
    "cdn.icon-icons.com", "icons8.com",
    "img.icons8.com", "iconscout.com",
    "fontawesome.com", "bootstrapcdn.com",
    "gravatar.com", "wp-includes/images",
    "s.w.org/images",
]

# URL path segments that indicate an icon/UI image
_ICON_URL_PATTERNS = [
    r"/icon[s/]", r"/logo[s/]", r"/avatar[s/]", r"/badge[s/]",
    r"/button[s/]", r"/sprite", r"/toolbar", r"/nav",
    r"/emoji", r"/emoticon", r"/thumbnail",
    r"icon\.", r"logo\.", r"avatar\.", r"badge\.",
    r"-icon\.", r"-logo\.", r"_icon\.", r"_logo\.",
]

# Alt text values that mean "this is a UI element, not content"
_ICON_ALT_PATTERNS = [
    r"^$",                        # no alt text
    r"^\d+$",                     # just a number ("2", "4" = notification count)
    r"^(linkedin|twitter|facebook|instagram|youtube|tiktok)$",
    r"\bicon\b", r"\blogo\b", r"\bavatar\b", r"\bbadge\b",
    r"\bbutton\b", r"\bsmiley\b", r"\bemoji\b",
    r"^(reply|forward|delete|archive|star|like|share)$",
]

# Tracking pixel patterns
_TRACKING_PATTERNS = [
    r"track\.", r"pixel\.", r"beacon\.", r"open\.php",
    r"analytics\.", r"trk\.", r"1x1", r"spacer",
    r"&open=", r"action=open", r"eo=", r"/o\.gif",
]


# ── text cleaning ─────────────────────────────────────────────────────────────

def _is_boilerplate_line(line: str) -> bool:
    l = line.lower().strip()
    if len(l) < 15:
        return True
    return any(phrase in l for phrase in _BOILERPLATE)


def clean_body(text: str) -> str:
    """Remove boilerplate lines and normalize whitespace."""
    if not text:
        return ""

    clean_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            clean_lines.append("")
            continue
        if _is_boilerplate_line(stripped):
            continue
        if re.fullmatch(r"https?://\S+", stripped):
            continue
        clean_lines.append(stripped)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(clean_lines)).strip()


# ── Stage 1 — heuristic image filter ─────────────────────────────────────────

def _is_tracking_image(src: str, tag) -> bool:
    w = str(tag.get("width",  ""))
    h = str(tag.get("height", ""))
    if w in ("0", "1") or h in ("0", "1"):
        return True
    src_lower = src.lower()
    return any(re.search(p, src_lower) for p in _TRACKING_PATTERNS)


def _is_icon_by_heuristic(src: str, alt: str, tag) -> bool:
    """Return True if this image is almost certainly a UI icon, not content."""
    src_lower = src.lower()
    alt_lower = alt.lower().strip()

    # Known icon CDNs
    if any(cdn in src_lower for cdn in _ICON_CDNS):
        return True

    # URL path patterns
    if any(re.search(p, src_lower) for p in _ICON_URL_PATTERNS):
        return True

    # Alt text patterns
    if any(re.search(p, alt_lower) for p in _ICON_ALT_PATTERNS):
        return True

    # Small declared size (UI icons are usually < 100px)
    for attr in ("width", "height"):
        val = tag.get(attr, "")
        if val and str(val).isdigit() and int(val) < 100:
            return True

    return False


def _heuristic_filter(candidates: list[dict]) -> list[dict]:
    """Keep only images that pass all heuristic checks."""
    return [c for c in candidates if not c["is_icon"]]


# ── Stage 2 — AI image filter ────────────────────────────────────────────────

def _ai_filter_images(candidates: list[dict], subject: str) -> list[str]:
    """
    Ask qwen to decide which candidate images are real article/content images
    vs UI elements, based on the newsletter subject + URL + alt text.
    Returns a list of approved URLs.
    """
    if not candidates:
        return []

    lines = "\n".join(
        f"{i+1}. URL: {c['src'][:120]}  |  Alt: \"{c['alt']}\""
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are filtering images extracted from a newsletter email.
Newsletter subject: "{subject}"

Below are images found in this email (URL and alt text).
Your job: identify which are REAL CONTENT images (article photos, product screenshots, charts, event images, people in the news) vs UI/icon images (logos, social media icons, notification badges, navigation buttons, email template decorations).

{lines}

Reply with ONLY the numbers of the CONTENT images, comma-separated.
If ALL are icons/UI elements, reply: none
If ALL are content images, reply: all

Examples:
- LinkedIn logo, Twitter bird icon, notification badge "2" → none
- OpenAI product screenshot, chart showing AI adoption → all
- Mix of article photo + LinkedIn icon → just the number of the article photo

Reply:"""

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":   OLLAMA_IMAGE_FILTER_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0, "num_predict": 50},
            },
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json()["response"].strip().lower()

        if "none" in raw:
            return []
        if "all" in raw:
            return [c["src"] for c in candidates]

        # Parse comma-separated numbers
        numbers = [int(x.strip()) for x in re.findall(r"\d+", raw)]
        return [
            candidates[n - 1]["src"]
            for n in numbers
            if 1 <= n <= len(candidates)
        ]

    except Exception:
        # AI unavailable — return all heuristic-passed candidates
        return [c["src"] for c in candidates]


# ── main extractor ────────────────────────────────────────────────────────────

def _extract_images_from_text(body: str) -> list[dict]:
    """
    Fallback: extract image URLs embedded in plain-text bodies.
    Beehiiv and similar senders write 'View image: (URL)' or markdown ![](URL)
    when the plain-text part is used instead of HTML.
    """
    candidates = []
    seen       = set()

    patterns = [
        r"View image:\s*\(?(https://[^\s\)\n]+)",       # Beehiiv plain text
        r"!\[([^\]]*)\]\((https://[^\s\)]+)\)",          # Markdown image
        r"(https://[^\s\)>\"'\n]+\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s\)>\"'\n]*)?)",
        r"(https://media\.beehiiv\.com/[^\s\)\n>\"']+)", # Beehiiv CDN
        r"(https://substackcdn\.com/image/[^\s\)\n>\"']+)",
        r"(https://cdn\.sanity\.io/[^\s\)\n>\"']+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, body, re.IGNORECASE):
            groups = match.groups()
            # For markdown, group[1] is the URL; others group[0]
            src = groups[-1].strip().rstrip(")")
            alt = groups[0].strip() if len(groups) > 1 else ""

            if not src.startswith("http") or src in seen:
                continue
            if any(re.search(p, src.lower()) for p in _TRACKING_PATTERNS):
                continue
            if any(cdn in src.lower() for cdn in _ICON_CDNS):
                continue

            seen.add(src)
            candidates.append({"src": src, "alt": alt, "is_icon": False})

    return candidates


def extract_images(html: str, subject: str = "", body_text: str = "") -> list[str]:
    """
    Two-stage image extraction:
      Stage 1 — heuristic: filter tracking pixels, icon CDNs, alt text, size
      Stage 2 — AI: phi4-mini validates remaining candidates against the subject

    Falls back to plain-text URL extraction when HTML yields nothing
    (e.g. Beehiiv newsletters that embed image URLs in plain-text parts).
    Returns up to 6 content-relevant image URLs.
    """
    candidates = []
    seen       = set()

    # ── parse HTML ────────────────────────────────────────────────────────────
    if html:
        soup = BeautifulSoup(html, "lxml")
        for img in soup.find_all("img"):
            src = img.get("src", "").strip()
            alt = img.get("alt", "").strip()
            if not src or src.startswith("data:") or not src.startswith("http"):
                continue
            if src in seen or _is_tracking_image(src, img):
                continue
            seen.add(src)
            candidates.append({
                "src":     src,
                "alt":     alt,
                "is_icon": _is_icon_by_heuristic(src, alt, img),
            })

    # ── fallback: extract from plain-text body ────────────────────────────────
    if not candidates and body_text:
        candidates = _extract_images_from_text(body_text)

    # Stage 1 — drop obvious icons
    passed = _heuristic_filter(candidates)
    if not passed:
        return []

    # Stage 2 — AI validation
    if subject and passed:
        approved = _ai_filter_images(passed, subject)
    else:
        approved = [c["src"] for c in passed]

    return approved[:6]


# ── per-email processing ──────────────────────────────────────────────────────

def process_email(email_data: dict, html: str = "") -> dict:
    """Add clean_body and images fields to an email dict."""
    result  = dict(email_data)
    subject = email_data.get("subject", "")
    result["clean_body"] = clean_body(email_data.get("body", ""))
    result["images"]     = (
        extract_images(html, subject) if html
        else email_data.get("images", [])
    )
    return result


# ── batch processing ──────────────────────────────────────────────────────────

def _process_file(path: str) -> int:
    if not os.path.exists(path):
        print(f"  {path} not found — skipping")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        emails = json.load(f)

    updated = 0
    for e in emails:
        changed = False
        if "clean_body" not in e:
            e["clean_body"] = clean_body(e.get("body", ""))
            changed = True
        # Backfill images — use stored html_body if available, else plain-text fallback
        if not e.get("images"):
            html  = e.get("html_body", "")
            body  = e.get("body", "")
            subj  = e.get("subject", "")
            e["images"] = extract_images(html, subj, body)
            changed = True
        if changed:
            updated += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    return updated


if __name__ == "__main__":
    print("\n  Cleaner")
    print("  " + "-" * 50)
    for path in [JSON_FILE_ALL, JSON_FILE_TODAY]:
        n = _process_file(path)
        print(f"  {path:<28}  {n} email(s) updated")
    print()
