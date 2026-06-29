"""
Review Queue Admin UI
Simple command-line interface for reviewing pending cases
"""

import os
import sys
import cv2
import base64
import numpy as np
from datetime import datetime
from database import get_db
from config import DATASET_DIR


class ReviewQueueUI:
    """Command-line interface for reviewing pending cases"""
    
    def __init__(self):
        self.db = get_db()
        self.reviewer_name = "admin"
    
    def decode_image(self, image_base64: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        try:
            image_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
    
    def display_review(self, review: dict) -> bool:
        """Display a review item and get decision"""
        print("\n" + "=" * 60)
        print(f"📋 Review ID: {review['id']}")
        print("=" * 60)
        print(f"👤 Candidate: {review['candidate_name'] or 'Unknown'}")
        print(f"🎯 Confidence: {review['confidence']:.3f}")
        print(f"📅 Timestamp: {review['timestamp']}")
        print(f"📝 Notes: {review.get('review_notes', '')}")
        
        if review.get('face_image_base64'):
            img = self.decode_image(review['face_image_base64'])
            if img is not None:
                cv2.imshow(f"Review - ID: {review['id']}", img)
                cv2.waitKey(1)
        
        print("\n📌 Options:")
        print("  1. Approve - Mark as {review['candidate_name']}")
        print("  2. Reject - Mark as Unknown")
        print("  3. Skip - Keep for later")
        print("  4. Exit Review")
        
        while True:
            choice = input("\n👉 Enter choice (1-4): ").strip()
            
            if choice == '1':
                return self.handle_approve(review)
            elif choice == '2':
                return self.handle_reject(review)
            elif choice == '3':
                cv2.destroyAllWindows()
                return True  # Continue to next
            elif choice == '4':
                cv2.destroyAllWindows()
                return False  # Exit
            else:
                print("❌ Invalid choice. Try again.")
    
    def handle_approve(self, review: dict) -> bool:
        """Handle approval of a review"""
        print("\n✅ Approving review...")
        
        # Get correct name
        name = review['candidate_name'] or input("Enter the correct name: ").strip()
        
        if not name:
            print("❌ Name is required for approval.")
            return True
        
        # Add notes
        notes = input("Add notes (optional): ").strip()
        
        # Confirm
        print(f"\n📌 Confirm: Approve as '{name}'? (y/n)")
        if input().strip().lower() != 'y':
            print("❌ Approval cancelled.")
            return True
        
        # Approve
        if self.db.approve_review(review['id'], name, self.reviewer_name, notes):
            print(f"✅ Approved: {name}")
            # Add to dataset if not exists
            self._add_to_dataset(name, review)
        else:
            print("❌ Failed to approve.")
        
        cv2.destroyAllWindows()
        return True
    
    def handle_reject(self, review: dict) -> bool:
        """Handle rejection of a review"""
        print("\n❌ Rejecting review...")
        
        notes = input("Reason for rejection (optional): ").strip()
        
        if self.db.reject_review(review['id'], self.reviewer_name, notes):
            print("✅ Review rejected")
        else:
            print("❌ Failed to reject.")
        
        cv2.destroyAllWindows()
        return True
    
    def _add_to_dataset(self, name: str, review: dict):
        """Add approved face to dataset"""
        if not review.get('face_image_base64'):
            return
        
        # Create directory if not exists
        person_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Save image
        img = self.decode_image(review['face_image_base64'])
        if img is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_review_{review['id']}.jpg"
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, img)
            print(f"📁 Saved image to: {filepath}")
    
    def run(self):
        """Run the review queue interface"""
        print("\n" + "=" * 60)
        print("📋 MANUAL REVIEW QUEUE")
        print("=" * 60)
        
        # Show statistics
        stats = self.db.get_review_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Total: {stats.get('total', 0)}")
        print(f"   Pending: {stats.get('pending', 0)}")
        print(f"   Approved: {stats.get('approved', 0)}")
        print(f"   Rejected: {stats.get('rejected', 0)}")
        print(f"   Avg Confidence: {stats.get('avg_confidence', 0):.3f}")
        
        if stats.get('pending', 0) == 0:
            print("\n✅ No pending reviews!")
            return
        
        # Get pending reviews
        reviews = self.db.get_pending_reviews()
        print(f"\n📋 Found {len(reviews)} pending reviews")
        
        # Review each item
        for i, review in enumerate(reviews, 1):
            print(f"\n📌 Review {i}/{len(reviews)}")
            if not self.display_review(review):
                break
        
        print("\n✅ Review session completed!")
        
        # Show updated statistics
        stats = self.db.get_review_statistics()
        print(f"\n📊 Updated Statistics:")
        print(f"   Pending: {stats.get('pending', 0)}")
        print(f"   Approved: {stats.get('approved', 0)}")
        print(f"   Rejected: {stats.get('rejected', 0)}")


def run_review_queue():
    """Entry point for review queue"""
    ui = ReviewQueueUI()
    ui.run()


if __name__ == "__main__":
    run_review_queue()
