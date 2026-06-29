"""
Database Module for Attendance System
Handles all database operations using SQLite
"""

import sqlite3
import os
import pandas as pd
import base64
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager


class DatabaseManager:
    """Manages SQLite database operations for the attendance system"""
    
    def __init__(self, db_path: str = "attendance.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _initialize_database(self):
        """Create all tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # ── People Table ──────────────────────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ── Attendance Table ─────────────────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    session_date DATE NOT NULL,
                    first_seen DATETIME,
                    last_seen DATETIME,
                    total_seconds REAL DEFAULT 0,
                    morning_status TEXT,
                    morning_desc TEXT,
                    afternoon_status TEXT,
                    afternoon_desc TEXT,
                    quit_time_status TEXT,
                    quit_time_desc TEXT,
                    final_status TEXT DEFAULT 'Absent',
                    permission_morning_used BOOLEAN DEFAULT 0,
                    permission_afternoon_used BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES people(id),
                    UNIQUE(person_id, session_date)
                )
            ''')
            
            # ── Permissions Table ────────────────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    month TEXT NOT NULL,
                    remaining INTEGER DEFAULT 5,
                    used INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES people(id),
                    UNIQUE(person_id, month)
                )
            ''')
            
            # ── Contacts Table ───────────────────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS contacts (
                    person_id INTEGER PRIMARY KEY,
                    email TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES people(id)
                )
            ''')
            
            # ── Detection Logs Table (Analytics) ────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    time_window TEXT,
                    is_recognized BOOLEAN,
                    FOREIGN KEY (person_id) REFERENCES people(id)
                )
            ''')
            
            # ── Review Queue Table ───────────────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    candidate_name TEXT,
                    confidence REAL NOT NULL,
                    face_image_base64 TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    review_status TEXT DEFAULT 'pending',
                    review_notes TEXT,
                    reviewed_by TEXT,
                    reviewed_at DATETIME,
                    FOREIGN KEY (person_id) REFERENCES people(id)
                )
            ''')
            
            # ── Review Logs Table ────────────────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS review_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    notes TEXT,
                    performed_by TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (review_id) REFERENCES review_queue(id)
                )
            ''')
            
            # ── Analytics Summary Table ──────────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_detections INTEGER DEFAULT 0,
                    successful_recognitions INTEGER DEFAULT 0,
                    failed_recognitions INTEGER DEFAULT 0,
                    unknown_faces INTEGER DEFAULT 0,
                    review_queue_entries INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    min_confidence REAL DEFAULT 0,
                    max_confidence REAL DEFAULT 0,
                    total_attendance INTEGER DEFAULT 0,
                    full_attendance INTEGER DEFAULT 0,
                    half_attendance INTEGER DEFAULT 0,
                    absent INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            ''')
            
            # ── Daily Stats Table ────────────────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    detections INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    first_seen DATETIME,
                    last_seen DATETIME,
                    total_time_seconds REAL DEFAULT 0,
                    status TEXT DEFAULT 'Absent',
                    FOREIGN KEY (person_id) REFERENCES people(id),
                    UNIQUE(person_id, date)
                )
            ''')
            
            # ── Indexes for Performance ──────────────────────────────────────
            # Attendance indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_person ON attendance(person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(session_date)')
            
            # Permissions indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_permissions_person ON permissions(person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_permissions_month ON permissions(month)')
            
            # Detection logs indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_person ON detection_logs(person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_time ON detection_logs(timestamp)')
            
            # Review queue indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(review_status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_review_timestamp ON review_queue(timestamp)')
            
            # Analytics indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_date ON analytics_summary(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dailystats_person ON daily_stats(person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dailystats_date ON daily_stats(date)')
            
            conn.commit()
    
    # ── People Operations ──────────────────────────────────────────────────
    
    def add_person(self, name: str) -> int:
        """Add a new person to the database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR IGNORE INTO people (name) VALUES (?)',
                (name.strip(),)
            )
            conn.commit()
            
            cursor.execute('SELECT id FROM people WHERE name = ?', (name.strip(),))
            result = cursor.fetchone()
            return result['id'] if result else None
    
    def get_person_id(self, name: str) -> Optional[int]:
        """Get person ID by name"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM people WHERE name = ?', (name.strip(),))
            result = cursor.fetchone()
            return result['id'] if result else None
    
    def get_all_people(self) -> List[str]:
        """Get all person names"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM people ORDER BY name')
            results = cursor.fetchall()
            return [row['name'] for row in results]
    
    def get_people_with_contacts(self) -> Dict[str, str]:
        """Get all people with their email contacts"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.name, c.email 
                FROM people p
                LEFT JOIN contacts c ON p.id = c.person_id
                WHERE c.email IS NOT NULL
            ''')
            results = cursor.fetchall()
            return {row['name']: row['email'] for row in results}
    
    # ── Attendance Operations ──────────────────────────────────────────────
    
    def save_attendance(self, name: str, data: Dict) -> bool:
        """
        Save or update attendance record for a person
        
        Args:
            name: Person's name
            data: Dictionary containing attendance data
        """
        person_id = self.get_person_id(name)
        if person_id is None:
            person_id = self.add_person(name)
        
        session_date = datetime.now().strftime("%Y-%m-%d")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE person_id = ? AND session_date = ?
            ''', (person_id, session_date))
            
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute('''
                    UPDATE attendance SET
                        first_seen = COALESCE(first_seen, ?),
                        last_seen = ?,
                        total_seconds = ?,
                        morning_status = ?,
                        morning_desc = ?,
                        afternoon_status = ?,
                        afternoon_desc = ?,
                        quit_time_status = ?,
                        quit_time_desc = ?,
                        final_status = ?,
                        permission_morning_used = ?,
                        permission_afternoon_used = ?
                    WHERE id = ?
                ''', (
                    data.get('first_seen'),
                    data.get('last_seen'),
                    data.get('total_seconds', 0),
                    data.get('morning_status'),
                    data.get('morning_desc'),
                    data.get('afternoon_status'),
                    data.get('afternoon_desc'),
                    data.get('quit_time_status'),
                    data.get('quit_time_desc'),
                    data.get('final_status', 'Absent'),
                    data.get('permitted_morning', 0),
                    data.get('permitted_afternoon', 0),
                    existing['id']
                ))
            else:
                cursor.execute('''
                    INSERT INTO attendance (
                        person_id, session_date, first_seen, last_seen,
                        total_seconds, morning_status, morning_desc,
                        afternoon_status, afternoon_desc,
                        quit_time_status, quit_time_desc,
                        final_status, permission_morning_used,
                        permission_afternoon_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    person_id,
                    session_date,
                    data.get('first_seen'),
                    data.get('last_seen'),
                    data.get('total_seconds', 0),
                    data.get('morning_status'),
                    data.get('morning_desc'),
                    data.get('afternoon_status'),
                    data.get('afternoon_desc'),
                    data.get('quit_time_status'),
                    data.get('quit_time_desc'),
                    data.get('final_status', 'Absent'),
                    data.get('permitted_morning', 0),
                    data.get('permitted_afternoon', 0)
                ))
            
            conn.commit()
            return True
    
    def get_attendance_report(self, date: Optional[str] = None) -> pd.DataFrame:
        """Get attendance report for a specific date or all dates"""
        with self.get_connection() as conn:
            if date:
                query = '''
                    SELECT 
                        p.name,
                        a.session_date,
                        a.first_seen,
                        a.last_seen,
                        a.total_seconds,
                        a.morning_status,
                        a.morning_desc,
                        a.afternoon_status,
                        a.afternoon_desc,
                        a.quit_time_status,
                        a.quit_time_desc,
                        a.final_status,
                        a.permission_morning_used,
                        a.permission_afternoon_used
                    FROM attendance a
                    JOIN people p ON a.person_id = p.id
                    WHERE a.session_date = ?
                    ORDER BY p.name
                '''
                df = pd.read_sql_query(query, conn, params=(date,))
            else:
                query = '''
                    SELECT 
                        p.name,
                        a.session_date,
                        a.first_seen,
                        a.last_seen,
                        a.total_seconds,
                        a.morning_status,
                        a.morning_desc,
                        a.afternoon_status,
                        a.afternoon_desc,
                        a.quit_time_status,
                        a.quit_time_desc,
                        a.final_status,
                        a.permission_morning_used,
                        a.permission_afternoon_used
                    FROM attendance a
                    JOIN people p ON a.person_id = p.id
                    ORDER BY a.session_date DESC, p.name
                '''
                df = pd.read_sql_query(query, conn)
            
            return df
    
    # ── Permission Operations ──────────────────────────────────────────────
    
    def save_permission(self, name: str, month: str, remaining: int):
        """Save or update permission record"""
        person_id = self.get_person_id(name)
        if person_id is None:
            person_id = self.add_person(name)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO permissions (person_id, month, remaining)
                VALUES (?, ?, ?)
            ''', (person_id, month, remaining))
            
            conn.commit()
    
    def get_permission(self, name: str, month: str) -> Optional[int]:
        """Get remaining permissions for a person"""
        person_id = self.get_person_id(name)
        if person_id is None:
            return None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT remaining FROM permissions
                WHERE person_id = ? AND month = ?
            ''', (person_id, month))
            result = cursor.fetchone()
            return result['remaining'] if result else None
    
    def update_permission(self, name: str, month: str, remaining: int):
        """Update remaining permissions"""
        person_id = self.get_person_id(name)
        if person_id is None:
            person_id = self.add_person(name)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE permissions 
                SET remaining = ?, updated_at = CURRENT_TIMESTAMP
                WHERE person_id = ? AND month = ?
            ''', (remaining, person_id, month))
            
            if cursor.rowcount == 0:
                cursor.execute('''
                    INSERT INTO permissions (person_id, month, remaining)
                    VALUES (?, ?, ?)
                ''', (person_id, month, remaining))
            
            conn.commit()
    
    def get_all_permissions(self) -> Dict:
        """Get all permissions as dictionary"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.name, pe.month, pe.remaining
                FROM permissions pe
                JOIN people p ON pe.person_id = p.id
            ''')
            results = cursor.fetchall()
            
            permissions = {}
            for row in results:
                permissions[(row['name'], row['month'])] = row['remaining']
            
            return permissions
    
    # ── Contact Operations ──────────────────────────────────────────────────
    
    def save_contact(self, name: str, email: str) -> bool:
        """Save or update contact email"""
        person_id = self.get_person_id(name)
        if person_id is None:
            person_id = self.add_person(name)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO contacts (person_id, email)
                VALUES (?, ?)
            ''', (person_id, email.strip()))
            
            conn.commit()
            return True
    
    def get_all_contacts(self) -> Dict[str, str]:
        """Get all contacts as dictionary"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.name, c.email
                FROM contacts c
                JOIN people p ON c.person_id = p.id
            ''')
            results = cursor.fetchall()
            return {row['name']: row['email'] for row in results}
    
    # ── Detection Log Operations ──────────────────────────────────────────
    
    def log_detection(self, name: str, confidence: float, 
                      time_window: str, is_recognized: bool):
        """Log a detection event for analytics"""
        person_id = self.get_person_id(name)
        if person_id is None:
            person_id = self.add_person(name)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detection_logs (person_id, confidence, time_window, is_recognized)
                VALUES (?, ?, ?, ?)
            ''', (person_id, confidence, time_window, is_recognized))
            conn.commit()
    
    # ── Review Queue Operations ──────────────────────────────────────────
    
    def add_to_review_queue(self, name: str, confidence: float, 
                           face_image: Optional[np.ndarray] = None,
                           notes: str = "") -> int:
        """
        Add a borderline case to the review queue
        
        Args:
            name: Candidate name (could be guessed)
            confidence: Recognition confidence score
            face_image: Cropped face image (optional)
            notes: Additional notes
            
        Returns:
            review_id: ID of the created review record
        """
        person_id = self.get_person_id(name) if name else None
        
        # Convert image to base64 if provided
        image_base64 = None
        if face_image is not None:
            try:
                _, buffer = cv2.imencode('.jpg', face_image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception:
                pass
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO review_queue (
                    person_id, candidate_name, confidence, 
                    face_image_base64, review_status, review_notes
                ) VALUES (?, ?, ?, ?, 'pending', ?)
            ''', (person_id, name, confidence, image_base64, notes))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_pending_reviews(self, limit: int = 50) -> List[Dict]:
        """Get all pending review items"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    rq.id,
                    rq.candidate_name,
                    rq.confidence,
                    rq.timestamp,
                    rq.review_notes,
                    p.name as actual_name,
                    rq.face_image_base64
                FROM review_queue rq
                LEFT JOIN people p ON rq.person_id = p.id
                WHERE rq.review_status = 'pending'
                ORDER BY rq.timestamp ASC
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
    
    def get_all_reviews(self, status: Optional[str] = None) -> List[Dict]:
        """Get all review items, optionally filtered by status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute('''
                    SELECT 
                        rq.id,
                        rq.candidate_name,
                        rq.confidence,
                        rq.timestamp,
                        rq.review_status,
                        rq.review_notes,
                        rq.reviewed_by,
                        rq.reviewed_at,
                        p.name as actual_name
                    FROM review_queue rq
                    LEFT JOIN people p ON rq.person_id = p.id
                    WHERE rq.review_status = ?
                    ORDER BY rq.timestamp DESC
                ''', (status,))
            else:
                cursor.execute('''
                    SELECT 
                        rq.id,
                        rq.candidate_name,
                        rq.confidence,
                        rq.timestamp,
                        rq.review_status,
                        rq.review_notes,
                        rq.reviewed_by,
                        rq.reviewed_at,
                        p.name as actual_name
                    FROM review_queue rq
                    LEFT JOIN people p ON rq.person_id = p.id
                    ORDER BY rq.timestamp DESC
                ''')
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
    
    def approve_review(self, review_id: int, actual_name: str, 
                       reviewer: str = "admin", notes: str = "") -> bool:
        """
        Approve a review and mark attendance for the person
        
        Args:
            review_id: ID of the review record
            actual_name: Correct name of the person
            reviewer: Name of the reviewer
            notes: Review notes
        """
        person_id = self.get_person_id(actual_name)
        if person_id is None:
            person_id = self.add_person(actual_name)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE review_queue 
                SET review_status = 'approved',
                    reviewed_by = ?,
                    reviewed_at = CURRENT_TIMESTAMP,
                    review_notes = ?
                WHERE id = ?
            ''', (reviewer, notes, review_id))
            
            cursor.execute('''
                INSERT INTO review_logs (review_id, action, notes, performed_by)
                VALUES (?, 'approved', ?, ?)
            ''', (review_id, notes, reviewer))
            
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute('''
                INSERT OR REPLACE INTO attendance 
                (person_id, session_date, final_status)
                VALUES (?, ?, 'Present (Manual Review)')
            ''', (person_id, today))
            
            conn.commit()
            return True
    
    def reject_review(self, review_id: int, reviewer: str = "admin", 
                     notes: str = "") -> bool:
        """Reject a review"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE review_queue 
                SET review_status = 'rejected',
                    reviewed_by = ?,
                    reviewed_at = CURRENT_TIMESTAMP,
                    review_notes = ?
                WHERE id = ?
            ''', (reviewer, notes, review_id))
            
            cursor.execute('''
                INSERT INTO review_logs (review_id, action, notes, performed_by)
                VALUES (?, 'rejected', ?, ?)
            ''', (review_id, notes, reviewer))
            
            conn.commit()
            return True
    
    def get_review_statistics(self) -> Dict:
        """Get statistics about the review queue"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN review_status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN review_status = 'approved' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN review_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    AVG(confidence) as avg_confidence
                FROM review_queue
            ''')
            
            stats = cursor.fetchone()
            return dict(stats) if stats else {
                'total': 0, 'pending': 0, 'approved': 0, 'rejected': 0, 'avg_confidence': 0
            }
    
    # ── Analytics Operations ──────────────────────────────────────────────
    
    def update_analytics_summary(self, date: str, stats: Dict):
        """Update or insert analytics summary for a date"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO analytics_summary (
                    date, total_detections, successful_recognitions,
                    failed_recognitions, unknown_faces, review_queue_entries,
                    avg_confidence, min_confidence, max_confidence,
                    total_attendance, full_attendance, half_attendance, absent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date,
                stats.get('total_detections', 0),
                stats.get('successful_recognitions', 0),
                stats.get('failed_recognitions', 0),
                stats.get('unknown_faces', 0),
                stats.get('review_queue_entries', 0),
                stats.get('avg_confidence', 0),
                stats.get('min_confidence', 0),
                stats.get('max_confidence', 0),
                stats.get('total_attendance', 0),
                stats.get('full_attendance', 0),
                stats.get('half_attendance', 0),
                stats.get('absent', 0)
            ))
            
            conn.commit()
    
    def get_analytics_summary(self, date: Optional[str] = None) -> Dict:
        """Get analytics summary for a specific date"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM analytics_summary WHERE date = ?
            ''', (date,))
            result = cursor.fetchone()
            return dict(result) if result else {}
    
    # ── Export Operations (Backward Compatibility) ──────────────────────
    
    def export_attendance_to_csv(self, filepath: str):
        """Export attendance data to CSV (for backward compatibility)"""
        df = self.get_attendance_report()
        df.to_csv(filepath, index=False)
        print(f"✅ Exported attendance to {filepath}")
    
    def export_permissions_to_csv(self, filepath: str):
        """Export permissions data to CSV (for backward compatibility)"""
        permissions = self.get_all_permissions()
        df = pd.DataFrame([
            {"Name": name, "Month": month, "Remaining": remaining}
            for (name, month), remaining in permissions.items()
        ])
        df.to_csv(filepath, index=False)
        print(f"✅ Exported permissions to {filepath}")
    
    def export_contacts_to_csv(self, filepath: str):
        """Export contacts data to CSV (for backward compatibility)"""
        contacts = self.get_all_contacts()
        df = pd.DataFrame([
            {"Name": name, "Email": email}
            for name, email in contacts.items()
        ])
        df.to_csv(filepath, index=False)
        print(f"✅ Exported contacts to {filepath}")
    
    def export_all_to_csv(self, output_dir: str = "exports"):
        """Export all data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.export_attendance_to_csv(f"{output_dir}/attendance_{timestamp}.csv")
        self.export_permissions_to_csv(f"{output_dir}/permissions_{timestamp}.csv")
        self.export_contacts_to_csv(f"{output_dir}/contacts_{timestamp}.csv")
        
        print(f"✅ All exports saved to: {output_dir}/")
    
    # ── Migration from CSV ──────────────────────────────────────────────────
    
    def migrate_from_csv(self, attendance_csv: str, permissions_csv: str, contacts_csv: str):
        """Migrate data from CSV files to SQLite database"""
        print("🔄 Starting migration from CSV to SQLite...")
        
        # Migrate people and attendance
        if os.path.exists(attendance_csv):
            df = pd.read_csv(attendance_csv)
            for _, row in df.iterrows():
                name = row['Name']
                data = {
                    'first_seen': row.get('First Seen'),
                    'last_seen': row.get('Last Seen'),
                    'total_seconds': row.get('Total Seconds', 0),
                    'morning_status': row.get('Morning Status'),
                    'morning_desc': row.get('Morning Description'),
                    'afternoon_status': row.get('Afternoon Status'),
                    'afternoon_desc': row.get('Afternoon Description'),
                    'quit_time_status': row.get('Quit Time Status'),
                    'quit_time_desc': row.get('Quit Time Description'),
                    'final_status': row.get('Final Status', 'Absent'),
                    'permitted_morning': row.get('Permission Morning Used', 0),
                    'permitted_afternoon': row.get('Permission Afternoon Used', 0)
                }
                self.save_attendance(name, data)
            print(f"✅ Migrated {len(df)} attendance records")
        
        # Migrate permissions
        if os.path.exists(permissions_csv):
            df = pd.read_csv(permissions_csv)
            for _, row in df.iterrows():
                name = row['Name']
                month = row['Month']
                remaining = row['Remaining']
                self.save_permission(name, month, remaining)
            print(f"✅ Migrated {len(df)} permission records")
        
        # Migrate contacts
        if os.path.exists(contacts_csv):
            df = pd.read_csv(contacts_csv)
            for _, row in df.iterrows():
                name = row['Name']
                email = row['Email']
                self.save_contact(name, email)
            print(f"✅ Migrated {len(df)} contact records")
        
        print("✅ Migration completed successfully!")


# ── Singleton instance ──────────────────────────────────────────────────
_db_instance = None

def get_db():
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
