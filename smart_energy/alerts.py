"""Optional push notifications for new events: Telegram and/or e-mail.

Configured through environment variables (see config.py); does nothing while none are set.
"""
import logging
import smtplib
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import requests

from . import config

log = logging.getLogger("alerts")
TUNIS = timezone(timedelta(hours=1))  # UTC+1, no DST


def format_event(event: dict) -> str:
    when = datetime.fromtimestamp(event["ts"], TUNIS).strftime("%Y-%m-%d %H:%M:%S")
    text = f"⚡ Smart Energy AI: {event['kind']} on {event['device_id']}\n{when} (Tunis time)"
    if event.get("power"):
        text += f" · {event['power']:,.0f} W"
    if event.get("detail"):
        text += f"\n{event['detail']}"
    return text


def notify(event: dict) -> None:
    """Send the event from a background thread, so a slow network never blocks ingestion."""
    if config.ALERT_KINDS and event["kind"] not in config.ALERT_KINDS:
        return
    telegram = config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID
    email = config.SMTP_HOST and config.ALERT_EMAIL_TO
    if telegram or email:
        threading.Thread(target=_send, args=(format_event(event),), daemon=True).start()


def _send(text: str) -> None:
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        try:
            r = requests.post(f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                              json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text}, timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            # don't log the exception text: it contains the URL, and so the bot token
            log.error("Telegram alert failed (%s)", getattr(e.response, "status_code", None) or type(e).__name__)

    if config.SMTP_HOST and config.ALERT_EMAIL_TO:
        msg = EmailMessage()
        msg["Subject"] = text.splitlines()[0]
        msg["From"] = config.ALERT_EMAIL_FROM or config.ALERT_EMAIL_TO
        msg["To"] = config.ALERT_EMAIL_TO
        msg.set_content(text)
        try:
            smtp_cls = smtplib.SMTP_SSL if config.SMTP_PORT == 465 else smtplib.SMTP
            with smtp_cls(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as smtp:
                if config.SMTP_PORT == 587:
                    smtp.starttls()
                if config.SMTP_USER:
                    smtp.login(config.SMTP_USER, config.SMTP_PASSWORD or "")
                smtp.send_message(msg)
        except (OSError, smtplib.SMTPException) as e:
            log.error("E-mail alert failed: %s", e)
