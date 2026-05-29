"""
auth.py — User authentication layer (signup / login).

Uses bcrypt directly for password hashing (avoids passlib/bcrypt version conflicts).
All user data is scoped by user_id from the users table.
"""

import bcrypt
import db


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def signup(name: str, email: str, password: str) -> dict:
    """
    Create a new user. Returns user dict on success.
    If a migrated account exists for this email, sets the password instead.
    Raises ValueError if a fully registered account already exists.
    """
    existing = db.get_user_by_email(email)
    if existing:
        if existing["password_hash"] == "migrated_no_password":
            # Activate the migrated account by setting a real password
            with db.get_cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash = %s, name = %s WHERE email = %s",
                    (hash_password(password), name, email),
                )
            return db.get_user_by_email(email)
        raise ValueError(f"Email already registered: {email}")
    return db.create_user(name, email, hash_password(password))


def login(email: str, password: str) -> dict:
    """
    Verify credentials. Returns user dict on success.
    Raises ValueError on bad email or wrong password.
    """
    user = db.get_user_by_email(email)
    if not user:
        raise ValueError("No account found with that email.")
    # Migration users have a plaintext sentinel — reject password auth for them
    if user["password_hash"] == "migrated_no_password":
        raise ValueError("This account was migrated. Please set a password via sign up.")
    if not verify_password(password, user["password_hash"]):
        raise ValueError("Incorrect password.")
    db.update_last_login(user["id"])
    return user


def get_user(user_id: int) -> dict | None:
    return db.get_user_by_id(user_id)
