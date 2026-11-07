"""Sends e-mail through any SMTP server (Gmail, Outlook, your school server...). Standard library only.

With SMTP_HOST empty (development) nothing is sent: the message, including the reset link, is printed in the server console instead.
"""
import logging
import smtplib
import ssl
from email.message import EmailMessage
from html import escape

from app.core.config import settings

log = logging.getLogger("uvicorn.error")


def send_email(to: str, subject: str, text: str, html: str | None = None) -> bool:
    if not settings.SMTP_HOST:
        log.warning("SMTP_HOST is not set, so no e-mail was sent. Message for %s:\n  Subject: %s\n%s", to, subject, text)
        return False
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = settings.SMTP_FROM or settings.SMTP_USER, to, subject
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    try:
        context = ssl.create_default_context()
        if settings.SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15, context=context)
        else:
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15)
            if settings.SMTP_STARTTLS:
                server.starttls(context=context)
        with server:
            if settings.SMTP_USER:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception:  # never let a mail problem break the request
        log.exception("Could not send e-mail to %s", to)
        return False


def send_reset_email(to: str, name: str, link: str, minutes: int) -> bool:
    first = (name or "there").split()[0]
    text = (f"Hi {first},\n\nWe received a request to reset your AnswerSheet AI password.\n"
            f"Open this link to choose a new one (valid for {minutes} minutes):\n\n{link}\n\n"
            "If you did not ask for this, you can ignore this e-mail. Your password will stay the same.\n")
    html = f"""<div style="font-family:Nunito,Arial,sans-serif;background:#E6F8FD;padding:28px">
<div style="max-width:480px;margin:auto;background:#fff;border-radius:16px;overflow:hidden">
<div style="background:#4A2C8F;color:#fff;padding:18px 24px;font-size:20px;font-weight:800">AnswerSheet <span style="color:#F7C535">AI</span></div>
<div style="padding:24px;color:#10105A;line-height:1.55">
<p style="margin:0 0 12px">Hi {escape(first)},</p>
<p style="margin:0 0 20px">We received a request to reset your password. This link works for {minutes} minutes.</p>
<p style="margin:0 0 22px"><a href="{escape(link)}" style="background:#F7C535;color:#10105A;text-decoration:none;font-weight:800;padding:12px 22px;border-radius:10px;display:inline-block">Choose a new password</a></p>
<p style="margin:0;color:#5B5B8A;font-size:13px">If you did not ask for this, ignore this e-mail and your password stays the same.</p>
</div></div></div>"""
    return send_email(to, "Reset your AnswerSheet AI password", text, html)
