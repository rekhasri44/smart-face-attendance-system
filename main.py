# main.py
import cv2
import pandas as pd
import collections
from datetime import datetime, time, timedelta

from config import (
    ATTENDANCE_CSV, FRAME_SKIP, CONSEC_DETECTS_REQUIRED,
    ABSENT_TIMEOUT, MIN_FACE_SIZE, MAX_PEOPLE_PER_FRAME,
    MORNING_EARLY_START, MORNING_PERMISSION_END,
    AFTERNOON_NORMAL_START, AFTERNOON_PERMISSION_END,
    AFTERNOON_PERMISSION_START, MORNING_PERMISSION_START,
    QUIT_TIME_START, QUIT_TIME_END
)
from recognition import build_embedding_db, recognize_face
from attendance import (
    load_permissions, save_permissions, mark_permission,
    bucket_morning, bucket_afternoon, bucket_quit_time, overall_status
)
from email_service import notify_all
from utils import draw_label, safe_to_csv, FaceStabilizer

# Cooldown constant
ATTENDANCE_COOLDOWN_SECONDS = 10


def try_open_webcam():
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
        if cap.isOpened():
            print(f"✅ Webcam opened on index {idx}.")
            return cap
        cap.release()
    print("❌ Webcam not accessible.")
    return None


def init_attendance(people):
    return {
        name: {
            "time_spans": [], "total_seconds": 0, "detections": [],
            "morning_first_seen": None, "afternoon_first_seen": None,
            "quit_time_seen": None, "morning": {}, "afternoon": {},
            "quit_time": {}, "final_status": "Absent",
            "permitted_morning": False, "permitted_afternoon": False,
            "detected_frames": 0, "session_start_time": None,
            "current_session_duration": 0.0
        }
        for name in people
    }


def draw_telemetry(frame, fps, total_faces, recognized, unregistered):
    """Draws a lightweight telemetry HUD in the top-left corner."""
    lines = [
        f"FPS: {fps:.1f}",
        f"Faces: {total_faces}",
        f"Recognized: {recognized}",
        f"Unregistered: {unregistered}",
    ]
    x, y_start, line_height = 10, 20, 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 1

    panel_w, panel_h = 180, line_height * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        y = y_start + i * line_height
        cv2.putText(frame, line, (x, y), font, scale, (0, 255, 180), thickness, cv2.LINE_AA)


def draw_activity_panel(frame, activity_log):
    """Renders recent detection history on the right side of the frame."""
    if not activity_log:
        return

    frame_h, frame_w = frame.shape[:2]
    panel_w = 220
    line_height = 22
    padding = 8
    panel_h = len(activity_log) * line_height + padding * 2
    x_start = frame_w - panel_w - 10
    y_start = 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x_start - 5, y_start),
                  (x_start + panel_w, y_start + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, (ts, label, color) in enumerate(reversed(activity_log)):
        time_str = ts.strftime("%H:%M:%S")
        text = f"{label}  {time_str}"
        y = y_start + padding + i * line_height
        cv2.putText(frame, text,
                    (x_start, y + line_height - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, color, 1, cv2.LINE_AA)


# ── Quality Validation ─────────────────────────────────────────────────────
QUALITY_MIN_FACE_SIZE = 70       # px — below this = too far (laptop-friendly)
QUALITY_EDGE_MARGIN   = 10       # px — face touching frame border
QUALITY_MIN_ASPECT    = 0.6      # w/h ratio — too narrow = turned/occluded
QUALITY_MAX_ASPECT    = 1.6      # w/h ratio — too wide = unusual angle

def check_face_quality(x, y, w, h, frame_w, frame_h):
    """
    Returns (passed: bool, reason: str | None).
    Geometry-only — no ML cost.
    """
    # Too small / too far from camera
    if w < QUALITY_MIN_FACE_SIZE or h < QUALITY_MIN_FACE_SIZE:
        return False, "Face Too Far"

    # Partially outside frame (clipped by edge)
    if (x <= QUALITY_EDGE_MARGIN or
        y <= QUALITY_EDGE_MARGIN or
        x + w >= frame_w - QUALITY_EDGE_MARGIN or
        y + h >= frame_h - QUALITY_EDGE_MARGIN):
        return False, "Adjust Position"

    # Skewed aspect ratio — turned head / partial occlusion
    aspect = w / h
    if not (QUALITY_MIN_ASPECT <= aspect <= QUALITY_MAX_ASPECT):
        return False, "Face Not Properly Visible"

    return True, None
# ───────────────────────────────────────────────────────────────────────────


def run_attendance():
    print("📷 Opening webcam...")
    cap = try_open_webcam()
    if cap is None:
        return

    # Get frame dimensions for quality validation
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐 Frame dimensions: {frame_w}x{frame_h}")

    webcam_start = datetime.now().strftime("%H:%M:%S")

    try:
        db, people = build_embedding_db()
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return

    if not db or not people:
        print("❌ No embeddings loaded.")
        return

    attendance = init_attendance(people)
    permissions = load_permissions(people)
    last_seen = {name: None for name in people}
    in_session = {name: False for name in people}
    interval_open = {name: None for name in people}
    consecutive_detects = {name: 0 for name in people}
    cooldown_until = {name: None for name in people}
    
    # Activity panel initialization
    activity_log = collections.deque(maxlen=5)
    last_activity_time = {}
    ACTIVITY_COOLDOWN = 3
    
    stabilizers = {}
    
    # Telemetry initialization
    frame_times = collections.deque(maxlen=30)
    fps = 0.0
    frame_total_faces = 0
    frame_recognized = 0
    frame_unregistered = 0
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cv2.namedWindow('Attendance - Press q to quit', cv2.WINDOW_NORMAL)
    frame_count = 0

    while True:
        # FPS Tracking
        frame_times.append(datetime.now().timestamp())
        if len(frame_times) >= 2:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read failed.")
            break

        display = frame.copy()
        now = datetime.now()
        now_t = now.time()

        if frame_count % FRAME_SKIP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)[:MAX_PEOPLE_PER_FRAME]
            names_already_assigned = set()
            
            # Reset per-frame counters
            frame_total_faces = 0
            frame_recognized = 0
            frame_unregistered = 0

            for idx, (x, y, w, h) in enumerate(faces):
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue
                
                frame_total_faces += 1
                
                # ── Quality Validation ─────────────────────────────────────
                quality_ok, quality_reason = check_face_quality(x, y, w, h, frame_w, frame_h)
                
                if not quality_ok:
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 165, 255), 2)
                    draw_label(display, quality_reason, x, y, (0, 165, 255))
                    frame_unregistered += 1
                    continue
                # ───────────────────────────────────────────────────────────
                
                face_img = frame[y:y+h, x:x+w]
                try:
                    face_img_resized = cv2.resize(face_img, (160, 160))
                except Exception:
                    continue

                raw_name, score = recognize_face(face_img_resized, db)
                
                # Counters
                if raw_name is not None and raw_name not in ("Unknown", "Unregistered Face"):
                    frame_recognized += 1
                else:
                    frame_unregistered += 1
                    
                    # Activity log for unregistered faces (with spam prevention)
                    if "__unknown__" not in last_activity_time or \
                       (now - last_activity_time["__unknown__"]).total_seconds() > ACTIVITY_COOLDOWN:
                        activity_log.append((now, "⚠ Unregistered", (0, 165, 255)))
                        last_activity_time["__unknown__"] = now

                # Position-based stabilizer
                slot_key = f"face_{idx}"
                
                if slot_key not in stabilizers:
                    stabilizers[slot_key] = FaceStabilizer()
                
                stable_label = stabilizers[slot_key].update(raw_name)

                if stable_label not in ("Detecting...", "Unregistered Face", "Unknown", None):
                    name = stable_label
                    color = (0, 255, 0)
                    duration = attendance[name]["current_session_duration"]
                    label_text = f"{name} ({score:.2f}) {duration:.1f}s"
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                    draw_label(display, label_text, x, y, color)

                    # ── Cooldown overlay (UI-only) ─────────────────────────
                    if cooldown_until[name] and now < cooldown_until[name]:
                        remaining = int((cooldown_until[name] - now).total_seconds()) + 1
                        draw_label(
                            display,
                            f"Attendance Confirmed ✓ ({remaining}s)",
                            x,
                            y + h + 20,
                            (0, 220, 120)
                        )
                    # ───────────────────────────────────────────────────────
                    
                    # Activity log for recognized faces (with spam prevention)
                    if cooldown_until.get(name) and now < cooldown_until.get(name):
                        if name not in last_activity_time or \
                           (now - last_activity_time[name]).total_seconds() > ACTIVITY_COOLDOWN:
                            activity_log.append((now, f"✓ {name}", (0, 255, 180)))
                            last_activity_time[name] = now

                    if name not in names_already_assigned:
                        names_already_assigned.add(name)
                        consecutive_detects[name] += 1

                        if consecutive_detects[name] >= CONSEC_DETECTS_REQUIRED:
                            record = attendance[name]
                            if not in_session[name]:
                                interval_open[name] = now
                                in_session[name] = True
                                record["session_start_time"] = now
                                
                                # Cooldown trigger (UI-only)
                                if cooldown_until[name] is None or now >= cooldown_until[name]:
                                    cooldown_until[name] = now + timedelta(seconds=ATTENDANCE_COOLDOWN_SECONDS)
                                
                            last_seen[name] = now
                            attendance[name]["detected_frames"] += 1
                            record["detections"].append(now)

                            # Time-window bucketing
                            if MORNING_EARLY_START <= now_t < MORNING_PERMISSION_END:
                                if record["morning_first_seen"] is None:
                                    record["morning_first_seen"] = now
                                    if MORNING_PERMISSION_START <= now_t < MORNING_PERMISSION_END:
                                        if mark_permission(name, permissions):
                                            record["permitted_morning"] = True
                            elif AFTERNOON_NORMAL_START <= now_t < AFTERNOON_PERMISSION_END:
                                if record["afternoon_first_seen"] is None:
                                    record["afternoon_first_seen"] = now
                                    if AFTERNOON_PERMISSION_START <= now_t < AFTERNOON_PERMISSION_END:
                                        if mark_permission(name, permissions):
                                            record["permitted_afternoon"] = True
                            elif QUIT_TIME_START <= now_t < QUIT_TIME_END:
                                if record["quit_time_seen"] is None:
                                    record["quit_time_seen"] = now

                            if record["session_start_time"]:
                                duration = (now - record["session_start_time"]).total_seconds()
                                record["current_session_duration"] = duration

                elif stable_label in ("Detecting...", "Unregistered Face", "Unknown"):
                    if stable_label == "Detecting...":
                        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 200, 0), 2)
                        draw_label(display, "Detecting...", x, y, (255, 200, 0))
                    else:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        draw_label(display, stable_label, x, y, (0, 0, 255))

            # Reset non-detected counters
            for name in people:
                if name not in names_already_assigned:
                    consecutive_detects[name] = 0

            # Session timeout
            for name in people:
                if in_session[name] and last_seen[name]:
                    if (now - last_seen[name]).total_seconds() > ABSENT_TIMEOUT:
                        t_in, t_out = interval_open[name], last_seen[name]
                        if t_in and t_out and t_out > t_in:
                            dur = (t_out - t_in).total_seconds()
                            attendance[name]["time_spans"].append((t_in, t_out))
                            attendance[name]["total_seconds"] += dur
                        in_session[name] = False
                        interval_open[name] = None
                        attendance[name]["session_start_time"] = None
                        attendance[name]["current_session_duration"] = 0.0

        frame_count += 1
        
        # Draw overlays
        draw_telemetry(display, fps, frame_total_faces, frame_recognized, frame_unregistered)
        draw_activity_panel(display, activity_log)
        
        cv2.imshow('Attendance - Press q to quit', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exited by user.")
            break

    webcam_end = datetime.now().strftime("%H:%M:%S")
    cap.release()
    cv2.destroyAllWindows()

    # Finalize sessions
    now = datetime.now()
    for name in people:
        if in_session[name] and interval_open[name]:
            t_in, t_out = interval_open[name], last_seen[name] or now
            if t_out > t_in:
                attendance[name]["time_spans"].append((t_in, t_out))
                attendance[name]["total_seconds"] += (t_out - t_in).total_seconds()

        rec = attendance[name]
        rec["morning"] = bucket_morning(rec["morning_first_seen"], rec["permitted_morning"])
        rec["afternoon"] = bucket_afternoon(rec["afternoon_first_seen"], rec["permitted_afternoon"])
        rec["quit_time"] = bucket_quit_time(rec["quit_time_seen"])
        rec["final_status"] = overall_status(rec) if rec["detected_frames"] > 0 else "Absent"

    # Save intermediate log
    rows = [{
        "Name": name,
        "Total Seconds": round(attendance[name]["total_seconds"], 2),
        "Morning Status": attendance[name]["morning"]["status"],
        "Morning Description": attendance[name]["morning"]["desc"],
        "Afternoon Status": attendance[name]["afternoon"]["status"],
        "Afternoon Description": attendance[name]["afternoon"]["desc"],
        "Quit Time Status": attendance[name]["quit_time"]["status"],
        "Quit Time Description": attendance[name]["quit_time"]["desc"],
        "Final Status": attendance[name]["final_status"],
        "Permission Morning Used": attendance[name]["permitted_morning"],
        "Permission Afternoon Used": attendance[name]["permitted_afternoon"],
    } for name in people]
    safe_to_csv(pd.DataFrame(rows), ATTENDANCE_CSV)
    save_permissions(permissions)

    print(f"\n⏱️  Start: {webcam_start}  |  End: {webcam_end}")

    # Manual override
    print("\n🛠️  Manually grant Full Attendance? (y/n)")
    if input().strip().lower() == 'y':
        names_input = input("Names (comma-separated): ").strip()
        for name in names_input.split(','):
            name = name.strip()
            if name in attendance:
                attendance[name]['final_status'] = 'Full Attendance (Manually Granted)'
                print(f"✅ Granted to {name}")
            else:
                print(f"❌ Not found: {name}")

    # Save final log with First/Last Seen
    final_rows = []
    for name in people:
        rec = attendance[name]
        dets = rec["detections"]
        first = min(dets).strftime("%Y-%m-%d %H:%M:%S") if dets else "Not Detected"
        last = max(dets).strftime("%Y-%m-%d %H:%M:%S") if dets else "Not Detected"
        final_rows.append({**rows[people.index(name)], "First Seen": first, "Last Seen": last})

    safe_to_csv(pd.DataFrame(final_rows), "final_attendance_log.csv")

    # Send email notifications
    notify_all(people, attendance)


if __name__ == "__main__":
    run_attendance()