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
    QUIT_TIME_START, QUIT_TIME_END,
    LIVENESS_ENABLED, EAR_THRESHOLD, BLINK_CONSECUTIVE_FRAMES,
    REQUIRED_BLINKS, LIVENESS_DETECTION_WINDOW,
    ENABLE_CHALLENGES, HEAD_MOVEMENT_THRESHOLD,
    CHALLENGE_TIMEOUT, CHALLENGES_REQUIRED,
    CAMERA_CONFIG, ACTIVE_CAMERAS, CAMERA_FRAME_SKIP, CAMERA_SWITCH_INTERVAL
)
from recognition import build_embedding_db, recognize_face
from attendance import (
    load_permissions, save_permissions, mark_permission,
    bucket_morning, bucket_afternoon, bucket_quit_time, overall_status
)
from email_service import notify_all
from utils import draw_label, safe_to_csv, FaceStabilizer
from liveness import BlinkDetector
from camera_manager import CameraManager
from tracker import FaceTracker  # ADDED: Import FaceTracker

# Try to import database module - handle gracefully if not available
try:
    from database import get_db
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("⚠️  Database module not found - review queue disabled")

# Cooldown constant
ATTENDANCE_COOLDOWN_SECONDS = 10
DUPLICATE_TIMEOUT = 60  # seconds to prevent duplicate markings

def init_attendance(people):
    return {
        name: {
            "time_spans": [],
            "total_seconds": 0,
            "detections": [],
            "morning_first_seen": None,
            "afternoon_first_seen": None,
            "quit_time_seen": None,
            "morning": {},
            "afternoon": {},
            "quit_time": {},
            "final_status": "Absent",
            "permitted_morning": False,
            "permitted_afternoon": False,
            "detected_frames": 0,
            "session_start_time": None,
            "current_session_duration": 0.0
        }
        for name in people
    }

def draw_telemetry(frame, fps, total_faces, recognized, unregistered, review_count=0, camera_info=""):
    lines = [
        f"FPS: {fps:.1f}",
        f"Faces: {total_faces}",
        f"Recognized: {recognized}",
        f"Unregistered: {unregistered}",
        f"Review Queue: {review_count}"
    ]
    x, y_start, line_height = 10, 20, 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 1

    panel_w, panel_h = 220, line_height * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        y = y_start + i * line_height
        cv2.putText(frame, line, (x, y), font, scale, (0, 255, 180), thickness, cv2.LINE_AA)
    
    # Add camera info at bottom
    if camera_info:
        cv2.putText(frame, camera_info, (10, frame.shape[0] - 10), 
                   font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

def draw_activity_panel(frame, activity_log):
    if not activity_log:
        return

    frame_h, frame_w = frame.shape[:2]
    panel_w = 220
    line_height = 22
    padding = 8
    panel_h = len(activity_log) * line_height + padding * 2
    x_start = frame_w - panel_w - 10
    y_start = 10

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
QUALITY_MIN_FACE_SIZE = 70
QUALITY_EDGE_MARGIN   = 10
QUALITY_MIN_ASPECT    = 0.6
QUALITY_MAX_ASPECT    = 1.6

def check_face_quality(x, y, w, h, frame_w, frame_h):
    if w < QUALITY_MIN_FACE_SIZE or h < QUALITY_MIN_FACE_SIZE:
        return False, "Face Too Far"

    if (x <= QUALITY_EDGE_MARGIN or
        y <= QUALITY_EDGE_MARGIN or
        x + w >= frame_w - QUALITY_EDGE_MARGIN or
        y + h >= frame_h - QUALITY_EDGE_MARGIN):
        return False, "Adjust Position"

    aspect = w / h
    if not (QUALITY_MIN_ASPECT <= aspect <= QUALITY_MAX_ASPECT):
        return False, "Face Not Properly Visible"

    return True, None

def run_attendance():
    print("📷 Initializing cameras...")
    camera_manager = CameraManager(
        camera_config=CAMERA_CONFIG,
        active_cameras=ACTIVE_CAMERAS,
        frame_skip=CAMERA_FRAME_SKIP
    )

    if not camera_manager.cameras:
        print("❌ No cameras available.")
        return

    camera_manager.display_camera_info()
    
    # Get first frame to get dimensions
    first_frame, _, _ = camera_manager.get_next_frame()
    if first_frame is None:
        print("❌ Could not get frame from any camera.")
        camera_manager.release_all()
        return
    
    frame_h, frame_w = first_frame.shape[:2]
    print(f"📐 Frame dimensions: {frame_w}x{frame_h}")

    webcam_start = datetime.now().strftime("%H:%M:%S")

    try:
        embedding_db, people = build_embedding_db()
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        camera_manager.release_all()
        return

    if not embedding_db or not people:
        print("❌ No embeddings loaded.")
        camera_manager.release_all()
        return

    attendance = init_attendance(people)
    permissions = load_permissions(people)
    last_seen = {name: None for name in people}
    in_session = {name: False for name in people}
    interval_open = {name: None for name in people}
    consecutive_detects = {name: 0 for name in people}
    cooldown_until = {name: None for name in people}
    
    # ── Duplicate Detection Prevention Cache ──────────────────────────────
    marked_attendance_cache = {}  # {(name, date): timestamp}
    # ───────────────────────────────────────────────────────────────────────

    # ── Liveness Detection Initialization with Challenges ──────────────────
    if LIVENESS_ENABLED:
        blink_detector = BlinkDetector(
            ear_threshold=EAR_THRESHOLD,
            consecutive_frames=BLINK_CONSECUTIVE_FRAMES,
            required_blinks=REQUIRED_BLINKS,
            detection_window=LIVENESS_DETECTION_WINDOW,
            head_movement_threshold=HEAD_MOVEMENT_THRESHOLD,
            challenge_timeout=CHALLENGE_TIMEOUT
        )
        liveness_verified = {name: False for name in people}
        liveness_last_verified_time = {name: None for name in people}
        person_challenges_completed = {name: 0 for name in people}
        print("✅ Liveness detection with challenges enabled")
        print(f"🎯 Complete {CHALLENGES_REQUIRED} challenges (blink + head movement)")
    else:
        blink_detector = None
        liveness_verified = {name: True for name in people}
        liveness_last_verified_time = {name: datetime.now() for name in people}
        person_challenges_completed = {name: 0 for name in people}
        print("⚠️  Liveness detection disabled")
    # ───────────────────────────────────────────────────────────────────────

    # ── Database and Review Queue ──────────────────────────────────────────
    db_handler = None
    if DATABASE_AVAILABLE:
        try:
            db_handler = get_db()
            print("✅ Database connection established")
        except Exception as e:
            print(f"⚠️  Database connection failed: {e}")
            db_handler = None
    # ───────────────────────────────────────────────────────────────────────

    activity_log = collections.deque(maxlen=5)
    last_activity_time = {}
    ACTIVITY_COOLDOWN = 3

    # REPLACED: stabilizers = {} with face_tracker
    face_tracker = FaceTracker(max_age=5, min_hits=3, iou_threshold=0.3)
    track_id_to_name = {}
    track_attendance_status = {}
    track_liveness_status = {}  # Track liveness per track

    frame_times = collections.deque(maxlen=30)
    fps = 0.0
    frame_total_faces = 0
    frame_recognized = 0
    frame_unregistered = 0
    total_review_count = 0  # Track total reviews across all frames
    current_camera = ""

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cv2.namedWindow('Attendance - Press q to quit', cv2.WINDOW_NORMAL)
    frame_count = 0

    while True:
        frame_times.append(datetime.now().timestamp())
        if len(frame_times) >= 2:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        
        # Get frame from camera manager
        frame, cam_key, stats = camera_manager.get_next_frame()
        
        if frame is None:
            # Try switching camera if no frame
            if camera_manager.switch_camera():
                print(f"🔄 Switched to next camera")
                continue
            else:
                print("❌ No camera available, exiting...")
                break
        
        # Update current camera info
        current_camera = cam_key
        cam_fps = stats.get('fps', 0) if stats else 0

        display = frame.copy()
        now = datetime.now()
        now_t = now.time()

        # ── Liveness detection with challenges on full frame ──────────────
        if LIVENESS_ENABLED and blink_detector:
            # Process with challenges enabled
            live_status, display, challenge_text = blink_detector.process_frame(
                frame, 
                enable_challenges=ENABLE_CHALLENGES
            )
            
            # Display challenge text on frame
            if challenge_text and not live_status:
                # Draw challenge instruction at bottom
                cv2.rectangle(display, (10, display.shape[0] - 70), 
                             (display.shape[1] - 10, display.shape[0] - 10),
                             (0, 0, 0), -1)
                cv2.putText(
                    display,
                    f"🎯 {challenge_text}",
                    (20, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            if live_status:
                # Update liveness for all active tracks
                for track_id in list(track_id_to_name.keys()):
                    name = track_id_to_name[track_id]
                    if name in people and name in liveness_verified:
                        if not liveness_verified[name]:
                            liveness_verified[name] = True
                            liveness_last_verified_time[name] = now
                            person_challenges_completed[name] = CHALLENGES_REQUIRED
                            activity_log.append((now, f"✅ Liveness Passed - {name}", (0, 255, 0)))
        # ───────────────────────────────────────────────────────────────────

        if frame_count % FRAME_SKIP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)[:MAX_PEOPLE_PER_FRAME]
            
            # Prepare detections and names for tracker
            detections = []
            detection_names = []
            detection_bboxes = []
            
            frame_total_faces = 0
            frame_recognized = 0
            frame_unregistered = 0
            frame_review_count = 0  # Reset per frame

            for (x, y, w, h) in faces:
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue
                
                frame_total_faces += 1
                
                quality_ok, quality_reason = check_face_quality(x, y, w, h, frame_w, frame_h)
                
                if not quality_ok:
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 165, 255), 2)
                    draw_label(display, quality_reason, x, y, (0, 165, 255))
                    frame_unregistered += 1
                    continue
                
                face_img = frame[y:y+h, x:x+w]
                try:
                    face_img_resized = cv2.resize(face_img, (160, 160))
                except Exception:
                    continue

                # ── Recognition with Review Queue ──────────────────────────────
                result = recognize_face(face_img_resized, embedding_db)
                if len(result) == 3:
                    raw_name, score, review_needed = result
                else:
                    raw_name, score = result
                    review_needed = False

                # ── Review Queue Logic ──────────────────────────────────────────
                if review_needed and db_handler is not None:
                    try:
                        # Store the face image for review
                        review_id = db_handler.add_to_review_queue(
                            name=raw_name if raw_name and raw_name not in ("Unknown", "Unregistered Face") else "Unknown",
                            confidence=score,
                            face_image=face_img,
                            notes=f"Borderline match - score: {score:.3f}"
                        )
                        
                        frame_review_count += 1
                        total_review_count += 1
                        
                        # Show on screen that review is needed
                        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 200, 0), 2)
                        draw_label(display, f"Review Needed: {score:.3f}", x, y, (255, 200, 0))
                        
                        # Log activity
                        if "review_needed" not in last_activity_time or \
                           (now - last_activity_time["review_needed"]).total_seconds() > ACTIVITY_COOLDOWN:
                            activity_log.append((now, f"📋 Review Needed - {raw_name or 'Unknown'}", (255, 200, 0)))
                            last_activity_time["review_needed"] = now
                        
                        # Don't mark attendance, skip to next face
                        continue
                    except Exception as e:
                        print(f"⚠️  Failed to add to review queue: {e}")
                        # Fall through to normal recognition
                # ──────────────────────────────────────────────────────────────────
                
                # Determine stable name
                if raw_name is not None and raw_name not in ("Unknown", "Unregistered Face"):
                    stable_label = raw_name
                    frame_recognized += 1
                else:
                    stable_label = "Unknown"
                    frame_unregistered += 1
                    
                    if "__unknown__" not in last_activity_time or \
                       (now - last_activity_time["__unknown__"]).total_seconds() > ACTIVITY_COOLDOWN:
                        activity_log.append((now, "⚠ Unregistered", (0, 165, 255)))
                        last_activity_time["__unknown__"] = now
                
                # Store detection for tracker
                detections.append([x, y, w, h])
                detection_names.append(stable_label)
                detection_bboxes.append((x, y, w, h))
            
            # Process detections through tracker
            if detections:
                confirmed_tracks = face_tracker.process_frame(detections, detection_names)
                
                # Process confirmed tracks
                for track_id, bbox, name in confirmed_tracks:
                    if name not in ("Unknown", "Unregistered Face", None):
                        x, y, w, h = bbox
                        
                        # Update track ID to name mapping
                        track_id_to_name[track_id] = name
                        
                        # Check liveness for this track
                        is_live = liveness_verified.get(name, False)
                        
                        # Draw tracked face
                        color = (0, 255, 0) if is_live else (255, 255, 0)
                        cv2.rectangle(display, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                        
                        duration = attendance[name]["current_session_duration"]
                        liveness_status = "✓" if is_live else "⏳"
                        
                        # Show challenge progress
                        if LIVENESS_ENABLED and not is_live:
                            completed = person_challenges_completed.get(name, 0)
                            liveness_status = f"🎯 {completed}/{CHALLENGES_REQUIRED}"
                        
                        label_text = f"ID:{track_id} {name} ({duration:.1f}s) {liveness_status}"
                        draw_label(display, label_text, int(x), int(y), color)
                        
                        # Track attendance status
                        if track_id not in track_attendance_status:
                            track_attendance_status[track_id] = {
                                'name': name, 
                                'attendance_marked': False,
                                'consecutive_count': 0
                            }
                        
                        # Increment consecutive detection count for this track
                        track_attendance_status[track_id]['consecutive_count'] += 1
                        
                        # Check cooldown
                        if cooldown_until[name] and now < cooldown_until[name]:
                            remaining = int((cooldown_until[name] - now).total_seconds()) + 1
                            draw_label(
                                display,
                                f"Attendance Confirmed ✓ ({remaining}s)",
                                int(x),
                                int(y) + h + 20,
                                (0, 220, 120)
                            )
                        
                        # Mark attendance if liveness verified and consecutive detection threshold met
                        if (track_attendance_status[track_id]['consecutive_count'] >= CONSEC_DETECTS_REQUIRED and 
                            is_live and 
                            not track_attendance_status[track_id]['attendance_marked']):
                            
                            # ── Duplicate Detection Prevention ──────────────────
                            attendance_key = (name, datetime.now().strftime("%Y-%m-%d"))
                            
                            if attendance_key in marked_attendance_cache:
                                last_mark = marked_attendance_cache[attendance_key]
                                if (datetime.now() - last_mark).total_seconds() < DUPLICATE_TIMEOUT:
                                    # Skip duplicate - already marked recently
                                    last_seen[name] = now
                                    continue
                            # ──────────────────────────────────────────────────────
                            
                            record = attendance[name]
                            if not in_session[name]:
                                interval_open[name] = now
                                in_session[name] = True
                                record["session_start_time"] = now
                                
                                if cooldown_until[name] is None or now >= cooldown_until[name]:
                                    cooldown_until[name] = now + timedelta(seconds=ATTENDANCE_COOLDOWN_SECONDS)
                            
                            # Mark attendance in cache
                            marked_attendance_cache[attendance_key] = now
                            track_attendance_status[track_id]['attendance_marked'] = True
                            
                            # Clean up old cache entries (older than 24 hours)
                            current_date = datetime.now().strftime("%Y-%m-%d")
                            for key in list(marked_attendance_cache.keys()):
                                if key[1] != current_date:
                                    del marked_attendance_cache[key]
                            
                            last_seen[name] = now
                            attendance[name]["detected_frames"] += 1
                            record["detections"].append(now)
                            
                            # Morning/Afternoon/Quit time logic
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
                            
                            # Log activity
                            if name not in last_activity_time or \
                               (now - last_activity_time[name]).total_seconds() > ACTIVITY_COOLDOWN:
                                status_icon = "✅" if is_live else "⏳"
                                activity_log.append((now, f"{status_icon} {name} marked", (0, 255, 180)))
                                last_activity_time[name] = now

            # Reset consecutive counts for tracks not seen
            for track_id in list(track_attendance_status.keys()):
                if track_id not in [t[0] for t in confirmed_tracks]:
                    if track_id in track_attendance_status:
                        track_attendance_status[track_id]['consecutive_count'] = 0

            # Reset liveness for people not seen recently
            for name in people:
                if name not in track_id_to_name.values():
                    # Check if this person was seen recently
                    if last_seen[name] and (now - last_seen[name]).total_seconds() > 5:
                        if LIVENESS_ENABLED and liveness_verified[name] and liveness_last_verified_time[name]:
                            if (now - liveness_last_verified_time[name]).total_seconds() > 30:
                                liveness_verified[name] = False
                                person_challenges_completed[name] = 0
                                if blink_detector:
                                    blink_detector.reset()

            # Close sessions for people not seen recently
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
        
        # Prepare camera info for telemetry
        camera_info = f"Camera: {current_camera} | Cam FPS: {cam_fps:.1f}"
        
        # Show total review count in telemetry
        draw_telemetry(display, fps, frame_total_faces, frame_recognized, frame_unregistered, total_review_count, camera_info)
        draw_activity_panel(display, activity_log)
        
        cv2.imshow('Attendance - Press q to quit', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exited by user.")
            break

    webcam_end = datetime.now().strftime("%H:%M:%S")
    camera_manager.release_all()
    cv2.destroyAllWindows()

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

    final_rows = []
    for name in people:
        rec = attendance[name]
        dets = rec["detections"]
        first = min(dets).strftime("%Y-%m-%d %H:%M:%S") if dets else "Not Detected"
        last = max(dets).strftime("%Y-%m-%d %H:%M:%S") if dets else "Not Detected"
        final_rows.append({**rows[people.index(name)], "First Seen": first, "Last Seen": last})

    safe_to_csv(pd.DataFrame(final_rows), "final_attendance_log.csv")
    notify_all(people, attendance)

if __name__ == "__main__":
    run_attendance()
