"""
migrate_json_to_pg.py — One-time migration of JSON files to PostgreSQL.

Migrates:
  - emails_all.json → emails, email_topics, email_images tables
  - digest_today.json → digests, digest_topics, digest_sections, digest_section_sources

Requires:
  - A user account in the DB (creates a default migration user if none exists)
  - DB schema already created (run: python db.py)

Usage:
  python migrate_json_to_pg.py
  python migrate_json_to_pg.py --email your@email.com  # associate emails with this user
"""

import json
import os
import sys
import re
import hashlib
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import db

JSON_FILE_ALL    = os.getenv("JSON_FILE_ALL",    "emails_all.json")
JSON_FILE_TODAY  = os.getenv("JSON_FILE_TODAY",  "emails_today.json")
DIGEST_FILE      = os.getenv("DIGEST_FILE",      "digest_today.json")
MIGRATION_EMAIL  = os.getenv("MIGRATION_EMAIL",  "migration@local")
MIGRATION_NAME   = os.getenv("MIGRATION_NAME",   "Migration User")


def _body_hash(body: str) -> str:
    normalized = re.sub(r"\s+", " ", body[:600]).strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()


def _get_or_create_user(email: str) -> int:
    user = db.get_user_by_email(email)
    if user:
        print(f"  Using existing user: {email} (id={user['id']})")
        return user["id"]

    # Create with a placeholder hash — real auth will update this
    user = db.create_user(MIGRATION_NAME, email, "migrated_no_password")
    print(f"  Created migration user: {email} (id={user['id']})")
    return user["id"]


def migrate_emails(user_id: int) -> dict:
    if not os.path.exists(JSON_FILE_ALL):
        print(f"  {JSON_FILE_ALL} not found — skipping email migration")
        return {"inserted": 0, "skipped": 0}

    with open(JSON_FILE_ALL, "r", encoding="utf-8") as f:
        emails = json.load(f)

    inserted = 0
    skipped  = 0

    print(f"  Migrating {len(emails)} emails from {JSON_FILE_ALL}...")

    for e in emails:
        body = e.get("body", "")
        body_hash = e.get("body_hash") or _body_hash(body)

        # Check if already exists
        if db.email_exists(user_id, body_hash):
            skipped += 1
            continue

        # Normalize topics — JSON might store as 'topic' (str) or 'topics' (list)
        topics = e.get("topics") or []
        if not topics and e.get("topic"):
            raw = e["topic"]
            topics = raw if isinstance(raw, list) else [raw]

        email_data = {
            "sender":       e.get("sender", ""),
            "subject":      e.get("subject", ""),
            "date":         e.get("date", ""),
            "body":         body,
            "html_body":    e.get("html_body", ""),
            "clean_body":   e.get("clean_body", ""),
            "has_html":     e.get("has_html", False),
            "body_hash":    body_hash,
            "category":     e.get("category", "unknown"),
            "fetched_at":   e.get("fetched_at", datetime.now().isoformat()),
            "topics":       topics,
            "images":       e.get("images", []),
        }

        email_id = db.save_email(user_id, email_data)
        if email_id:
            inserted += 1
        else:
            skipped += 1

    return {"inserted": inserted, "skipped": skipped}


def migrate_digest(user_id: int) -> bool:
    if not os.path.exists(DIGEST_FILE):
        print(f"  {DIGEST_FILE} not found — skipping digest migration")
        return False

    with open(DIGEST_FILE, "r", encoding="utf-8") as f:
        digest = json.load(f)

    date_str     = digest.get("date", str(datetime.now().date()))
    generated_at = digest.get("generated_at", datetime.now().isoformat())

    print(f"  Migrating digest for {date_str}...")

    digest_id = db.save_digest(user_id, {
        "date":         date_str,
        "generated_at": generated_at,
        "topics":       digest.get("topics", {}),
    })
    print(f"  Digest saved (id={digest_id}, {len(digest.get('topics', {}))} topics)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON data to PostgreSQL")
    parser.add_argument(
        "--email",
        default=MIGRATION_EMAIL,
        help="User email to associate migrated data with (default: migration@local)",
    )
    args = parser.parse_args()

    print("\n  Migration: JSON -> PostgreSQL")
    print("  " + "-" * 50)

    user_id = _get_or_create_user(args.email)

    print()
    result = migrate_emails(user_id)
    print(f"  Emails: {result['inserted']} inserted, {result['skipped']} skipped")

    print()
    migrate_digest(user_id)

    print()
    print("  Migration complete.")
    print(f"  User id={user_id} owns all migrated data.")
    print()


if __name__ == "__main__":
    main()
