"""
Migration script to convert CSV data to SQLite database
Run this once to migrate existing data
"""

import os
from database import DatabaseManager
from config import ATTENDANCE_CSV, PERMISSIONS_CSV, CONTACTS_CSV


def migrate():
    """Migrate data from CSV to SQLite"""
    print("=" * 60)
    print("📊 CSV to SQLite Migration Tool")
    print("=" * 60)
    
    # Check if CSV files exist
    files_exist = False
    if os.path.exists(ATTENDANCE_CSV):
        print(f"✅ Found: {ATTENDANCE_CSV}")
        files_exist = True
    else:
        print(f"❌ Not found: {ATTENDANCE_CSV}")
    
    if os.path.exists(PERMISSIONS_CSV):
        print(f"✅ Found: {PERMISSIONS_CSV}")
        files_exist = True
    else:
        print(f"❌ Not found: {PERMISSIONS_CSV}")
    
    if os.path.exists(CONTACTS_CSV):
        print(f"✅ Found: {CONTACTS_CSV}")
        files_exist = True
    else:
        print(f"❌ Not found: {CONTACTS_CSV}")
    
    if not files_exist:
        print("\n❌ No CSV files found. Nothing to migrate.")
        return
    
    # Check if database already exists
    if os.path.exists("attendance.db"):
        response = input("\n⚠️  Database already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return
    
    # Perform migration
    print("\n🔄 Starting migration...")
    db = DatabaseManager()
    db.migrate_from_csv(ATTENDANCE_CSV, PERMISSIONS_CSV, CONTACTS_CSV)
    
    print("\n✅ Migration completed successfully!")
    print(f"📁 Database saved as: attendance.db")
    
    # Show summary
    print("\n📊 Migration Summary:")
    conn = db.get_connection().__enter__()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM people")
    people_count = cursor.fetchone()[0]
    print(f"   - People: {people_count}")
    
    cursor.execute("SELECT COUNT(*) FROM attendance")
    attendance_count = cursor.fetchone()[0]
    print(f"   - Attendance records: {attendance_count}")
    
    cursor.execute("SELECT COUNT(*) FROM permissions")
    permission_count = cursor.fetchone()[0]
    print(f"   - Permission records: {permission_count}")
    
    cursor.execute("SELECT COUNT(*) FROM contacts")
    contact_count = cursor.fetchone()[0]
    print(f"   - Contact records: {contact_count}")
    
    conn.close()


if __name__ == "__main__":
    migrate()
