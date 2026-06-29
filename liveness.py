"""
Liveness Detection Module using MediaPipe Face Mesh
Detects blinks using Eye Aspect Ratio (EAR)
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from typing import Tuple


class BlinkDetector:
    """
    Detects blinks using Eye Aspect Ratio (EAR) calculation.
    Requires 1-2 blinks to confirm liveness.
    """
    
    # Eye landmark indices for MediaPipe Face Mesh
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    def __init__(
        self,
        ear_threshold: float = 0.23,
        consecutive_frames: int = 3,
        required_blinks: int = 1,
        detection_window: int = 10
    ):
        """
        Args:
            ear_threshold: EAR below this indicates eye closed
            consecutive_frames: Frames required for blink completion
            required_blinks: Number of blinks needed to verify liveness
            detection_window: Number of recent frames to track for blink detection
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.required_blinks = required_blinks
        self.detection_window = detection_window
        
        # State tracking
        self.ear_history = deque(maxlen=detection_window)
        self.blink_counter = 0
        self.frame_counter = 0
        self.eye_closed_frames = 0
        self.is_blinking = False
        self.liveness_verified = False
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def calculate_ear(self, landmarks, eye_indices) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        Where p1..p6 are eye landmark points
        """
        try:
            points = []
            for idx in eye_indices:
                landmark = landmarks.landmark[idx]
                points.append([landmark.x, landmark.y])
            
            points = np.array(points)
            
            d1 = np.linalg.norm(points[1] - points[5])
            d2 = np.linalg.norm(points[2] - points[4])
            d3 = np.linalg.norm(points[0] - points[3])
            
            if d3 == 0:
                return 0.0
                
            ear = (d1 + d2) / (2.0 * d3)
            return ear
        except Exception:
            return 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Process a single frame for blink detection.
        
        Args:
            frame: BGR image
            
        Returns:
            (liveness_verified, annotated_frame)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        
        if not results.multi_face_landmarks:
            return self.liveness_verified, annotated_frame
        
        landmarks = results.multi_face_landmarks[0]
        
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(avg_ear)
        self.frame_counter += 1
        
        if avg_ear < self.ear_threshold:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= self.consecutive_frames:
                self.blink_counter += 1
                self.is_blinking = True
                self.eye_closed_frames = 0
        
        if self.blink_counter >= self.required_blinks:
            self.liveness_verified = True
        
        # Display status
        if self.liveness_verified:
            status_text = "Live ✓"
            color = (0, 255, 0)
        elif self.blink_counter > 0:
            status_text = f"Blinking... ({self.blink_counter}/{self.required_blinks})"
            color = (0, 165, 255)
        else:
            status_text = "Please blink"
            color = (0, 0, 255)
        
        cv2.putText(
            annotated_frame,
            f"Liveness: {status_text}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )
        
        cv2.putText(
            annotated_frame,
            f"EAR: {avg_ear:.3f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        
        return self.liveness_verified, annotated_frame
    
    def reset(self):
        """Reset liveness detection state"""
        self.blink_counter = 0
        self.eye_closed_frames = 0
        self.is_blinking = False
        self.liveness_verified = False
        self.ear_history.clear()
        self.frame_counter = 0
    
    def is_live(self) -> bool:
        """Check if liveness has been verified"""
        return self.liveness_verified


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = BlinkDetector(required_blinks=1)
    
    print("Press 'q' to quit")
    print("Look at the camera and blink to verify liveness")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        verified, annotated = detector.process_frame(frame)
        cv2.imshow("Blink Detection Test", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
