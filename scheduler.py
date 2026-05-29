"""
scheduler.py — Automated fetch + digest pipeline.

Runs immediately when started, then repeats every REPEAT_HOURS hours (default 5).
Run once; it loops in the background until you stop it.

Usage:
    python scheduler.py                  # runs now, then every 5 hours
    REPEAT_HOURS=3 python scheduler.py   # custom interval

To run silently in background on Windows:
    start /B pythonw scheduler.py

To run immediately once and exit (for testing):
    python scheduler.py --now
"""

import os
import sys
import time
import schedule
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

REPEAT_HOURS = int(os.getenv("REPEAT_HOURS", "5"))


# ── helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _active_accounts() -> list[tuple[int, int]]:
    """Return (user_id, gmail_account_id) for every active Gmail connection."""
    import db
    with db.get_cursor() as cur:
        cur.execute(
            """SELECT u.id AS user_id, g.id AS gmail_account_id
               FROM users u
               JOIN gmail_accounts g ON g.user_id = u.id
               WHERE g.is_active = TRUE"""
        )
        return [(r["user_id"], r["gmail_account_id"]) for r in cur.fetchall()]


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline():
    """Fetch emails and generate digest for every active user."""
    import gmail_oauth
    import summarizer

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    _log(f"=== Pipeline starting ({stamp}) ===")

    accounts = _active_accounts()
    if not accounts:
        _log("No active users with Gmail connected — nothing to do.")
        return

    for user_id, gmail_account_id in accounts:
        _log(f"--- User {user_id} / account {gmail_account_id} ---")

        # Step 1 — fetch
        _log("Fetching emails from Gmail...")
        try:
            stats = gmail_oauth.run_fetch_pipeline(
                user_id,
                gmail_account_id=gmail_account_id,
                progress_cb=lambda msg: _log(msg),
            )
        except Exception as ex:
            _log(f"Fetch failed: {ex}")
            continue

        if "error" in stats:
            _log(f"Fetch error: {stats['error']}")
            continue

        _log(f"Fetch done — {stats['saved']} saved, {stats['skipped']} skipped, {stats['errors']} errors")

        # Step 2 — digest (only if there are new emails)
        if stats.get("saved", 0) > 0:
            _log("Generating digest...")
            try:
                result = summarizer.generate_daily_digest(user_id, gmail_account_id=gmail_account_id)
                n = len(result.get("topics", {}))
                _log(f"Digest done — {n} topic(s) generated")
            except Exception as ex:
                _log(f"Digest failed: {ex}")
        else:
            _log("No new emails — skipping digest generation.")

    _log(f"=== Pipeline complete. Next run in {REPEAT_HOURS} hour(s) ===\n")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--now" in sys.argv:
        _log("--now flag detected — running pipeline once and exiting.")
        run_pipeline()
        sys.exit(0)

    # Run once immediately, then repeat every REPEAT_HOURS hours
    _log(f"Scheduler started. Running now, then every {REPEAT_HOURS} hour(s).")
    _log("Press Ctrl+C to stop.\n")

    run_pipeline()
    schedule.every(REPEAT_HOURS).hours.do(run_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(45)
