# attendance.py
import os
import pandas as pd
from datetime import datetime, time
from config import (
    PERMISSIONS_CSV, PERMISSIONS_PER_MONTH,
    MORNING_EARLY_START, MORNING_EARLY_END, MORNING_NORMAL_END,
    MORNING_LATE_END, MORNING_PERMISSION_START, MORNING_PERMISSION_END,
    AFTERNOON_NORMAL_START, AFTERNOON_NORMAL_END, AFTERNOON_LATE_END,
    AFTERNOON_PERMISSION_START, AFTERNOON_PERMISSION_END,
    QUIT_TIME_START, QUIT_TIME_END
)
from utils import safe_to_csv


def get_month_key(date: datetime) -> str:
    return date.strftime("%Y-%m")


def load_permissions(people: list) -> dict:
    if os.path.exists(PERMISSIONS_CSV):
        df = pd.read_csv(PERMISSIONS_CSV, index_col=[0, 1])
        perms = {(name, month): count for (name, month), count in df['Remaining'].items()}
    else:
        perms = {}
    month_key = get_month_key(datetime.now())
    for name in people:
        if (name, month_key) not in perms:
            perms[(name, month_key)] = PERMISSIONS_PER_MONTH
    return perms


def save_permissions(permissions: dict):
    df = pd.DataFrame([
        {"Name": name, "Month": month, "Remaining": count}
        for (name, month), count in permissions.items()
    ])
    safe_to_csv(df, PERMISSIONS_CSV)


def mark_permission(name: str, permissions: dict) -> bool:
    key = (name, get_month_key(datetime.now()))
    if permissions.get(key, PERMISSIONS_PER_MONTH) > 0:
        permissions[key] -= 1
        print(f"✅ {name} used a permission. Remaining: {permissions[key]}")
        return True
    print(f"❌ {name} has no permissions left this month.")
    return False


def bucket_morning(arrival_time, permitted: bool) -> dict:
    if not arrival_time:
        return {"status": "Absent", "desc": "No Appearance"}
    t = arrival_time.time()
    if MORNING_EARLY_START <= t < MORNING_EARLY_END:
        return {"status": "Present", "desc": "Early"}
    elif MORNING_EARLY_END <= t < MORNING_NORMAL_END:
        return {"status": "Present", "desc": "Normal Timing"}
    elif MORNING_NORMAL_END <= t < MORNING_LATE_END:
        return {"status": "Absent", "desc": "Late"}
    elif MORNING_PERMISSION_START <= t < MORNING_PERMISSION_END:
        if permitted:
            return {"status": "Present", "desc": "Permission Used"}
        return {"status": "Absent", "desc": "No Permission"}
    return {"status": "Absent", "desc": "Out of Window"}


def bucket_afternoon(arrival_time, permitted: bool) -> dict:
    if not arrival_time:
        return {"status": "Absent", "desc": "No Appearance"}
    t = arrival_time.time()
    if AFTERNOON_NORMAL_START <= t < AFTERNOON_NORMAL_END:
        return {"status": "Present", "desc": "Normal Timing"}
    elif AFTERNOON_NORMAL_END <= t < AFTERNOON_LATE_END:
        return {"status": "Absent", "desc": "Late"}
    elif AFTERNOON_PERMISSION_START <= t < AFTERNOON_PERMISSION_END:
        if permitted:
            return {"status": "Present", "desc": "Permission Used"}
        return {"status": "Absent", "desc": "No Permission"}
    return {"status": "Absent", "desc": "Out of Window"}


def bucket_quit_time(arrival_time) -> dict:
    if not arrival_time:
        return {"status": "Absent", "desc": "No Appearance"}
    t = arrival_time.time()
    if QUIT_TIME_START <= t < QUIT_TIME_END:
        return {"status": "Present", "desc": "Quit Time Seen"}
    return {"status": "Absent", "desc": "Not Seen During Quit Time"}


def overall_status(record: dict) -> str:
    m = record.get('morning', {}).get('status', 'Absent')
    a = record.get('afternoon', {}).get('status', 'Absent')
    q = record.get('quit_time', {}).get('status', 'Absent')
    if m == "Present" and a == "Present":
        return "Full Attendance"
    elif q == "Present":
        return "Full Attendance (Quit Time)"
    elif m == "Present" or a == "Present":
        return "Half Attendance"
    return "Absent"
