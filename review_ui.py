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
           
