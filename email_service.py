# email_service.py
import os
import smtplib
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import SENDER_EMAIL, SENDER_PASSWORD, CONTACTS_CSV


def load_contacts() -> dict:
    """Load name->email map from contacts.csv, fallback to hardcoded."""
    if os.path.exists(CONTACTS_CSV):
        df = pd.read_csv(CONTACTS_CSV)
        return dict(zip(df['Name'].str.lower(), df['Email']))
    return {}  # Empty fallback; add entries to contacts.csv


def send_email(to_email: str, subject: str, body: str):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        server.quit()
        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Email failed for {to_email}: {e}")


def notify_all(people: list, attendance: dict):
    contacts = load_contacts()
    for name in people:
        email = contacts.get(name.lower())
        if email:
            status = attendance[name]['final_status']
            send_email(
                email,
                subject="Attendance Notification",
                body=f"Hello {name},\n\nYour attendance status today: {status}.\n\nRegards,\nAttendance System"
            )