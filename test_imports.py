"""
Test script to verify all imports work correctly
Run this to check if everything is set up properly
"""

print("=" * 60)
print("🔍 TESTING ALL IMPORTS")
print("=" * 60)

# Test 1: Config
print("\n1. Testing config...")
try:
    from config import *
    print("   ✅ Config imported successfully")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")

# Test 2: Database
print("\n2. Testing database...")
try:
    from database import get_db
    db = get_db()
    print("   ✅ Database imported successfully")
except Exception as e:
    print(f"   ❌ Database import failed: {e}")

# Test 3: Recognition
print("\n3. Testing recognition...")
try:
    from recognition import build_embedding_db, recognize_face
    print("   ✅ Recognition imported successfully")
except Exception as e:
    print(f"   ❌ Recognition import failed: {e}")

# Test 4: Attendance
print("\n4. Testing attendance...")
try:
    from attendance import load_permissions, save_permissions, mark_permission
    print("   ✅ Attendance imported successfully")
except Exception as e:
    print(f"   ❌ Attendance import failed: {e}")

# Test 5: Liveness
print("\n5. Testing liveness...")
try:
    from liveness import BlinkDetector
    print("   ✅ Liveness imported successfully")
except Exception as e:
    print(f"   ❌ Liveness import failed: {e}")

# Test 6: Cache Manager
print("\n6. Testing cache manager...")
try:
    from cache_manager import CacheManager
    print("   ✅ Cache Manager imported successfully")
except Exception as e:
    print(f"   ❌ Cache Manager import failed: {e}")

# Test 7: Camera Manager
print("\n7. Testing camera manager...")
try:
    from camera_manager import CameraManager
    print("   ✅ Camera Manager imported successfully")
except Exception as e:
    print(f"   ❌ Camera Manager import failed: {e}")

# Test 8: Tracker
print("\n8. Testing tracker...")
try:
    from tracker import FaceTracker, SortTracker
    print("   ✅ Tracker imported successfully")
except Exception as e:
    print(f"   ❌ Tracker import failed: {e}")

# Test 9: Analytics
print("\n9. Testing analytics...")
try:
    from analytics import AnalyticsEngine
    print("   ✅ Analytics imported successfully")
except Exception as e:
    print(f"   ❌ Analytics import failed: {e}")

# Test 10: Review UI
print("\n10. Testing review UI...")
try:
    from review_ui import ReviewQueueUI
    print("   ✅ Review UI imported successfully")
except Exception as e:
    print(f"   ❌ Review UI import failed: {e}")

# Test 11: Email Service
print("\n11. Testing email service...")
try:
    from email_service import notify_all, send_email
    print("   ✅ Email Service imported successfully")
except Exception as e:
    print(f"   ❌ Email Service import failed: {e}")

# Test 12: Utils
print("\n12. Testing utils...")
try:
    from utils import draw_label, safe_to_csv, FaceStabilizer
    print("   ✅ Utils imported successfully")
except Exception as e:
    print(f"   ❌ Utils import failed: {e}")

print("\n" + "=" * 60)
print("✅ IMPORT TEST COMPLETE")
print("=" * 60)
