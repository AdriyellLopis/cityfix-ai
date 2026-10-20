"""SMS notifications for CityFix.

Demo mode (default): messages are written to the notification log and shown in the app, nothing is sent.
Live mode: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM (Streamlit secrets or environment variables).
Another South African provider (Clickatell, BulkSMS, SMSPortal) can be swapped in by editing _send_live().
"""
import os
import re

import db


def normalise_phone(raw: str):
    """Return a South African mobile number as +27XXXXXXXXX, or None if it is not valid."""
    d = re.sub(r"[^\d+]", "", raw or "")
    if d.startswith("+27"):
        n = d[3:]
    elif d.startswith("0027"):
        n = d[4:]
    elif d.startswith("27") and len(d) == 11:
        n = d[2:]
    elif d.startswith("0"):
        n = d[1:]
    else:
        return None
    return "+27" + n if re.fullmatch(r"[6-8]\d{8}", n) else None


def mask_phone(phone: str) -> str:
    return phone[:3] + " ••• ••• " + phone[-3:] if phone and len(phone) >= 6 else ""


def live_enabled() -> bool:
    return all(os.environ.get(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"))


def _send_live(to: str, body: str):
    import requests
    sid = os.environ["TWILIO_ACCOUNT_SID"]
    resp = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                         data={"To": to, "From": os.environ["TWILIO_FROM"], "Body": body},
                         auth=(sid, os.environ["TWILIO_AUTH_TOKEN"]), timeout=10)
    resp.raise_for_status()


def send_sms(ticket: dict, body: str) -> str:
    """Send (or, in demo mode, record) an SMS for a ticket. Only sends if the resident gave consent."""
    if not ticket.get("phone") or not ticket.get("sms_consent"):
        return "skipped"
    status = "demo"
    if live_enabled():
        try:
            _send_live(ticket["phone"], body)
            status = "sent"
        except Exception:
            status = "failed"
    db.log_notification(ticket["id"], "sms", body, status)
    return status


def msg_received(ref: str, secret: str, urgency: str, sla: int) -> str:
    return f"CityFix: request {ref} received. Tracking code {secret}. Priority {urgency}, first reply within {sla} hrs."


def msg_update(ref: str, status: str, reply: str = "") -> str:
    base = f"CityFix {ref}: status {status}."
    return (base + " Officer: " + reply.strip()[:90]) if reply.strip() else base
