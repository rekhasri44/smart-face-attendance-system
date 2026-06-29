"""
Liveness Detection Module using MediaPipe Face Mesh
Detects blinks using Eye Aspect Ratio (EAR)
And head movement challenges for anti-spoofing
"""

import cv2
import mediapipe as mp
import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, List
from enum import Enum


class ChallengeType(Enum):
    """Types of liveness challenges"""
    BLINK = "blink"
    HEAD_LEFT = "turn_left"
    HEAD_RIGHT = "turn_right"
    HEAD_UP = "look_up"
    HEAD_DOWN = "look_down"


class BlinkDetector:
    """
    Detects blinks using Eye Aspect Ratio (EAR) calculation.
    Also handles head movement challenges.
    """
    
    # Eye landmark indices for MediaPipe Face Mesh
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # Nose tip index for head tracking
    NOSE_INDEX = 1
    
    def __init__(
        self,
        ear_threshold: float = 0.23,
        consecutive_frames: int = 3,
        required_blinks: int = 1,
        detection_window: int = 10,
        head_movement_threshold: float = 0.03,
        challenge_timeout: int = 5
    ):
        """
        Args:
            ear_threshold: EAR below this indicates eye closed
            consecutive_frames: Frames required for blink completion
            required_blinks: Number of blinks needed to verify liveness
            detection_window: Number of recent frames to track for blink detection
            head_movement_threshold: Minimum movement to detect head turn
            challenge_timeout: Seconds before challenge expires
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.required_blinks = required_blinks
        self.detection_window = detection_window
        self.head_movement_threshold = head_movement_threshold
        self.challenge_timeout = challenge_timeout
        
        # Blink detection state
        self.ear_history = deque(maxlen=detection_window)
        self.blink_counter = 0
        self.frame_counter = 0
        self.eye_closed_frames = 0
        self.is_blinking = False
        self.liveness_verified = False
        
        # Head movement challenge state
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_completed = False
        self.initial_nose_position = None
        self.movement_detected = False
        self.challenge_attempts = 0
        self.max_attempts = 3
        self.last_movement_direction = None
        
        # Challenge history
        self.completed_challenges = []
        self.failed_challenges = []
        
        # Available challenges
        self.challenges = [
            ChallengeType.BLINK,
            ChallengeType.HEAD_LEFT,
            ChallengeType.HEAD_RIGHT,
            ChallengeType.HEAD_UP,
            ChallengeType.HEAD_DOWN
        ]
        
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
        """Calculate Eye Aspect Ratio (EAR)"""
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
    
    def get_nose_position(self, landmarks) -> Optional[np.ndarray]:
        """Get nose tip position from landmarks"""
        try:
            nose = landmarks.landmark[self.NOSE_INDEX]
            return np.array([nose.x, nose.y])
        except Exception:
            return None
    
    def detect_head_movement(self, current_position) -> Tuple[bool, str]:
        """
        Detect head movement direction
        
        Returns:
            (movement_detected, direction)
            direction: 'left', 'right', 'up', 'down'
        """
        if self.initial_nose_position is None:
            self.initial_nose_position = current_position
            return False, ""
        
        # Calculate displacement
        displacement = current_position - self.initial_nose_position
        
        # Determine direction
        # Note: In camera view, moving head right = nose moves left (negative x)
        # We invert for intuitive direction
        dx = -displacement[0]  # Invert for intuitive direction
        dy = -displacement[1]  # Invert for intuitive direction
        
        # Check if movement exceeds threshold
        if abs(dx) > self.head_movement_threshold or abs(dy) > self.head_movement_threshold:
            if abs(dx) > abs(dy):
                # Horizontal movement
                if dx > 0:
                    self.last_movement_direction = "right"
                    return True, "right"
                else:
                    self.last_movement_direction = "left"
                    return True, "left"
            else:
                # Vertical movement
                if dy > 0:
                    self.last_movement_direction = "up"
                    return True, "up"
                else:
                    self.last_movement_direction = "down"
                    return True, "down"
        
        return False, ""
    
    def get_challenge_description(self, challenge_type: ChallengeType) -> str:
        """Get human-readable description of a challenge"""
        descriptions = {
            ChallengeType.BLINK: "Blink once",
            ChallengeType.HEAD_LEFT: "Turn your head left",
            ChallengeType.HEAD_RIGHT: "Turn your head right",
            ChallengeType.HEAD_UP: "Look up",
            ChallengeType.HEAD_DOWN: "Look down"
        }
        return descriptions.get(challenge_type, "Unknown challenge")
    
    def generate_challenge(self) -> ChallengeType:
        """Generate a random challenge that hasn't been completed yet"""
        available = [c for c in self.challenges if c not in self.completed_challenges]
        if not available:
            # All challenges completed
            return None
        
        return random.choice(available)
    
    def start_challenge(self):
        """Start a new challenge"""
        self.current_challenge = self.generate_challenge()
        if self.current_challenge:
            self.challenge_start_time = self.frame_counter
            self.challenge_completed = False
            self.initial_nose_position = None
            self.movement_detected = False
            self.challenge_attempts += 1
            
            # For blink challenge, reset blink counter
            if self.current_challenge == ChallengeType.BLINK:
                self.blink_counter = 0
                self.eye_closed_frames = 0
    
    def check_challenge(self) -> bool:
        """Check if current challenge is completed"""
        if self.current_challenge is None:
            return False
        
        # Check timeout
        if self.challenge_start_time is not None:
            elapsed = self.frame_counter - self.challenge_start_time
            if elapsed > self.challenge_timeout * 30:  # ~30 FPS
                return False
        
        # Check specific challenge types
        if self.current_challenge == ChallengeType.BLINK:
            if self.blink_counter >= self.required_blinks:
                self.challenge_completed = True
                return True
        
        elif self.current_challenge in [ChallengeType.HEAD_LEFT, ChallengeType.HEAD_RIGHT,
                                        ChallengeType.HEAD_UP, ChallengeType.HEAD_DOWN]:
            if self.movement_detected:
                # Verify correct direction
                expected = {
                    ChallengeType.HEAD_LEFT: "left",
                    ChallengeType.HEAD_RIGHT: "right",
                    ChallengeType.HEAD_UP: "up",
                    ChallengeType.HEAD_DOWN: "down"
                }
                if self.last_movement_direction == expected.get(self.current_challenge):
                    self.challenge_completed = True
                    return True
        
        return False
    
    def process_frame(self, frame: np.ndarray, enable_challenges: bool = True) -> Tuple[bool, np.ndarray, Optional[str]]:
        """
        Process a single frame for liveness detection with challenges.
        
        Args:
            frame: BGR image
            enable_challenges: Whether to enable head movement challenges
            
        Returns:
            (liveness_verified, annotated_frame, challenge_text)
            challenge_text: Current challenge description or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        challenge_text = None
        
        if not results.multi_face_landmarks:
            return self.liveness_verified, annotated_frame, challenge_text
        
        landmarks = results.multi_face_landmarks[0]
        
        # Calculate EAR
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(avg_ear)
        self.frame_counter += 1
        
        # Detect blinks
        if avg_ear < self.ear_threshold:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= self.consecutive_frames:
                self.blink_counter += 1
                self.is_blinking = True
                self.eye_closed_frames = 0
        
        # Head movement for challenges
        nose_position = self.get_nose_position(landmarks)
        if nose_position is not None and enable_challenges:
            movement_detected, direction = self.detect_head_movement(nose_position)
            if movement_detected:
                self.movement_detected = True
        
        # Check challenge if active
        if self.current_challenge is not None:
            completed = self.check_challenge()
            if completed:
                self.completed_challenges.append(self.current_challenge)
                self.liveness_verified = len(self.completed_challenges) >= 2
                
                # Generate next challenge if more needed
                if not self.liveness_verified:
                    self.start_challenge()
            
            challenge_text = self.get_challenge_description(self.current_challenge)
        
        # Start first challenge if not started and liveness not verified
        if self.current_challenge is None and not self.liveness_verified and enable_challenges:
            self.start_challenge()
            if self.current_challenge:
                challenge_text = self.get_challenge_description(self.current_challenge)
        
        # Update display
        self._update_display(annotated_frame, avg_ear, challenge_text)
        
        return self.liveness_verified, annotated_frame, challenge_text
    
    def _update_display(self, frame: np.ndarray, avg_ear: float, challenge_text: Optional[str]):
        """Update frame with liveness and challenge status"""
        y_position = 60
        
        # Liveness status
        if self.liveness_verified:
            status_text = "✅ Live Verified!"
            color = (0, 255, 0)
        else:
            status_text = f"⏳ Liveness: {len(self.completed_challenges)}/2 challenges"
            color = (0, 165, 255)
        
        cv2.putText(
            frame,
            status_text,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )
        y_position += 30
        
        # Challenge text
        if challenge_text and not self.liveness_verified:
            cv2.putText(
                frame,
                f"🎯 Challenge: {challenge_text}",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2,
                cv2.LINE_AA
            )
            y_position += 30
        
        # EAR value
        cv2.putText(
            frame,
            f"EAR: {avg_ear:.3f}",
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        y_position += 25
        
        # Completed challenges
        if self.completed_challenges:
            completed_text = f"✅ Completed: {len(self.completed_challenges)}/2"
            cv2.putText(
                frame,
                completed_text,
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
    
    def reset(self):
        """Reset all liveness detection state"""
        self.blink_counter = 0
        self.eye_closed_frames = 0
        self.is_blinking = False
        self.liveness_verified = False
        self.ear_history.clear()
        self.frame_counter = 0
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_completed = False
        self.initial_nose_position = None
        self.movement_detected = False
        self.challenge_attempts = 0
        self.completed_challenges = []
        self.failed_challenges = []
    
    def is_live(self) -> bool:
        """Check if liveness has been verified"""
        return self.liveness_verified


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = BlinkDetector(required_blinks=1)
    
    print("Press 'q' to quit")
    print("Complete 2 challenges to verify liveness")
    print("Challenges: Blink, Turn head left/right, Look up/down")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        verified, annotated, challenge = detector.process_frame(frame)
        
        cv2.imshow("Liveness Detection with Challenges", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
