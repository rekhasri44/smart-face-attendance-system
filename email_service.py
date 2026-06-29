import os
import smtplib
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import SENDER_EMAIL, SENDER_PASSWORD, CONTACTS_CSV, USE_DATABASE

# Try to import database module
if USE_DATABASE:
    try:
        from database import get_db
        db = get_db()
    except ImportError:
        USE_DATABASE = False


def load_contacts() -> dict:
    """Load name->email map from database or CSV"""
    if USE_DATABASE:
        try:
            db = get_db()
            return db.get_all_contacts()
        except Exception:
            # Fallback to CSV
            return load_contacts_from_csv()
    else:
        return load_contacts_from_csv()


def load_contacts_from_csv() -> dict:
    """Load contacts from CSV (backward compatibility)"""
    if os.path.exists(CONTACTS_CSV):
        df = pd.read_csv(CONTACTS_CSV)
        return dict(zip(df['Name'].str.lower(), df['Email']))
    return {}


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
