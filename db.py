"""CityFix ticket store (SQLite). Tickets, two-way message threads and officer corrections."""
import os
import random
import sqlite3
import string
from datetime import datetime

import pandas as pd

DB_PATH = os.environ.get("CITYFIX_DB", "cityfix.db")


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def init():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ref TEXT UNIQUE, secret TEXT,
            created TEXT, updated TEXT, text TEXT, area TEXT,
            category TEXT, urgency TEXT, department TEXT, sla_hours INTEGER,
            confidence REAL, needs_review INTEGER, status TEXT, awaiting_officer INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER, sender TEXT, body TEXT, created TEXT);
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER, channel TEXT, body TEXT, status TEXT, created TEXT);
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER, text TEXT,
            old_category TEXT, new_category TEXT, old_urgency TEXT, new_urgency TEXT, created TEXT);
        """)
        have = {r[1] for r in c.execute("PRAGMA table_info(tickets)")}
        for name, ddl in [("address", "TEXT"), ("phone", "TEXT"), ("sms_consent", "INTEGER DEFAULT 0")]:
            if name not in have:
                c.execute(f"ALTER TABLE tickets ADD COLUMN {name} {ddl}")


def create_ticket(text: str, area: str, r: dict, created: str = None, status: str = "Open",
                  address: str = None, phone: str = None, sms_consent: bool = False):
    """r is the dict returned by router.triage. Returns (ref, secret)."""
    secret = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    created = created or now()
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO tickets (secret, created, updated, text, area, category, urgency, department,
               sla_hours, confidence, needs_review, status, awaiting_officer, address, phone, sms_consent)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?)""",
            (secret, created, created, text, area, r["category"], r["urgency"], r["department"], r["sla_hours"],
             r["confidence"], int(r["needs_human_review"]), "Under review" if r["needs_human_review"] and status == "Open" else status,
             address, phone, int(bool(sms_consent))))
        tid = cur.lastrowid
        ref = f"CF-{1000 + tid}"
        c.execute("UPDATE tickets SET ref=? WHERE id=?", (ref, tid))
    return ref, secret


def get_ticket(ref: str, secret: str = None):
    with _conn() as c:
        row = c.execute("SELECT * FROM tickets WHERE ref=?", (ref.strip().upper(),)).fetchone()
    if row is None:
        return None
    if secret is not None and row["secret"] != secret.strip().upper():
        return None
    return dict(row)


def list_tickets() -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql_query("SELECT * FROM tickets ORDER BY id DESC", c)


def count_tickets() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def add_message(ticket_id: int, sender: str, body: str, created: str = None):
    """sender: 'resident', 'system' or 'officer'. Resident replies flag the ticket as awaiting an officer."""
    created = created or now()
    with _conn() as c:
        c.execute("INSERT INTO messages (ticket_id, sender, body, created) VALUES (?,?,?,?)",
                  (ticket_id, sender, body, created))
        awaiting = 1 if sender == "resident" else (0 if sender == "officer" else None)
        if awaiting is not None:
            c.execute("UPDATE tickets SET awaiting_officer=?, updated=? WHERE id=?", (awaiting, created, ticket_id))
        else:
            c.execute("UPDATE tickets SET updated=? WHERE id=?", (created, ticket_id))
        if sender == "resident":  # a resident reply reopens a resolved ticket
            c.execute("UPDATE tickets SET status='Open' WHERE id=? AND status='Resolved'", (ticket_id,))


def get_messages(ticket_id: int):
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM messages WHERE ticket_id=? ORDER BY id", (ticket_id,))]


ALLOWED = {"category", "urgency", "department", "sla_hours", "status", "needs_review", "awaiting_officer"}


def update_ticket(ticket_id: int, **fields):
    fields = {k: v for k, v in fields.items() if k in ALLOWED}
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields) + ", updated=?"
    with _conn() as c:
        c.execute(f"UPDATE tickets SET {sets} WHERE id=?", (*fields.values(), now(), ticket_id))


def log_correction(ticket_id, text, old_cat, new_cat, old_urg, new_urg):
    with _conn() as c:
        c.execute("INSERT INTO corrections (ticket_id, text, old_category, new_category, old_urgency, new_urgency, created)"
                  " VALUES (?,?,?,?,?,?,?)", (ticket_id, text, old_cat, new_cat, old_urg, new_urg, now()))


def get_corrections() -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql_query("SELECT * FROM corrections ORDER BY id", c)


def log_notification(ticket_id: int, channel: str, body: str, status: str):
    with _conn() as c:
        c.execute("INSERT INTO notifications (ticket_id, channel, body, status, created) VALUES (?,?,?,?,?)",
                  (ticket_id, channel, body, status, now()))


def get_notifications(ticket_id: int):
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM notifications WHERE ticket_id=? ORDER BY id", (ticket_id,))]
