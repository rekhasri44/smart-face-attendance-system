"""
Face Tracking Module using SORT (Simple Online and Realtime Tracking)
Combines Kalman filters with Hungarian algorithm for robust tracking
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes
    Tracks state: [x, y, w, h, vx, vy, vw, vh]
    where (x, y) is center, (w, h) is size, v is velocity
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize tracker with bounding box
        
        Args:
            bbox: [x, y, w, h] in pixel coordinates
        """
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we measure x, y, w, h)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[0:4, 0:4] *= 0.1
        
        # Process noise
        self.kf.Q[0:4, 0:4] *= 0.01
        self.kf.Q[4:8, 4:8] *= 0.1
        
        # Initial state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        # Track variables
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Store history
        self.history = []
    
    def _convert_bbox_to_z(self, bbox):
        """
        Convert bounding box to state vector
        
        Args:
            bbox: [x, y, w, h]
            
        Returns:
            State vector [x, y, w, h]
        """
        x, y, w, h = bbox
        return np.array([x, y, w, h]).reshape(4, 1)
    
    def _convert_z_to_bbox(self, z):
        """
        Convert state vector to bounding box
        
        Args:
            z: State vector [x, y, w, h]
            
        Returns:
            Bounding box [x, y, w, h]
        """
        return z.reshape(4,)
    
    def update(self, bbox):
        """
        Update tracker with new measurement
        
        Args:
            bbox: [x, y, w, h] in pixel coordinates
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Update Kalman filter with measurement
        z = self._convert_bbox_to_z(bbox)
        self.kf.update(z)
    
    def predict(self):
        """
        Predict next state of the track
        
        Returns:
            Predicted bounding box [x, y, w, h]
        """
        if self.kf.x[4] + self.kf.x[5] <= 0:
            self.kf.x[4] *= 0.0
            self.kf.x[5] *= 0.0
        
        # Predict next state
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        
        # Return predicted bounding box
        return self._convert_z_to_bbox(self.kf.x[:4])
    
    def get_state(self):
        """
        Get current state
        
        Returns:
            Current bounding box [x, y, w, h]
        """
        return self._convert_z_to_bbox(self.kf.x[:4])


class SortTracker:
    """
    SORT Tracker - Simple Online and Realtime Tracking
    Tracks multiple objects with Kalman filters and Hungarian algorithm
    """
    
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker
        
        Args:
            max_age: Maximum frames without update before track is deleted
            min_hits: Minimum hits required to confirm a track
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.track_id_to_name = {}  # Map track ID to person name
        self.track_name_history = {}  # Track ID to list of (name, timestamp)
        
        # Reset counter
        KalmanBoxTracker.count = 0
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IOU) between two boxes
        
        Args:
            box1: [x1, y1, x2, y2] (top-left, bottom-right)
            box2: [x1, y1, x2, y2] (top-left, bottom-right)
            
        Returns:
            IOU value between 0 and 1
        """
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def _match_detections_to_trackers(self, detections, trackers):
        """
        Match detections to existing trackers using Hungarian algorithm
        
        Args:
            detections: List of bounding boxes [x, y, w, h]
            trackers: List of KalmanBoxTracker objects
            
        Returns:
            matched_indices: List of (detection_idx, tracker_idx) tuples
            unmatched_detections: List of detection indices
            unmatched_trackers: List of tracker indices
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
        
        # Calculate IOU between all detections and trackers
        iou_matrix = np.zeros((len(detections), len(trackers)))
        
        for d, detection in enumerate(detections):
            for t, tracker in enumerate(trackers):
                predicted_bbox = tracker.predict()
                iou_matrix[d, t] = self._calculate_iou(detection, predicted_bbox)
        
        # Hungarian algorithm for optimal matching
        matched_indices = []
        
        if min(iou_matrix.shape) > 0:
            # Hungarian algorithm minimizes cost, so we use negative IOU
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            for row, col in zip(row_ind, col_ind):
                if iou_matrix[row, col] >= self.iou_threshold:
                    matched_indices.append((row, col))
        
        # Find unmatched detections and trackers
        matched_detections = [m[0] for m in matched_indices]
        matched_trackers = [m[1] for m in matched_indices]
        
        unmatched_detections = [d for d in range(len(detections)) 
                               if d not in matched_detections]
        unmatched_trackers = [t for t in range(len(trackers)) 
                             if t not in matched_trackers]
        
        return matched_indices, unmatched_detections, unmatched_trackers
    
    def update(self, detections, detection_names=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of bounding boxes [x, y, w, h]
            detection_names: List of names for each detection (optional)
            
        Returns:
            List of (track_id, bbox, name) tuples for confirmed tracks
        """
        self.frame_count += 1
        
        # Predict new positions for existing trackers
        predicted_boxes = []
        for tracker in self.trackers:
            predicted_boxes.append(tracker.predict())
        
        # Match detections to trackers
        matched, unmatched_detections, unmatched_trackers = \
            self._match_detections_to_trackers(detections, self.trackers)
        
        # Update matched trackers
        for detection_idx, tracker_idx in matched:
            self.trackers[tracker_idx].update(detections[detection_idx])
            
            # Update name mapping if name provided
            if detection_names and detection_idx < len(detection_names):
                track_id = self.trackers[tracker_idx].id
                name = detection_names[detection_idx]
                
                if name not in ("Unknown", "Unregistered Face", None):
                    self.track_id_to_name[track_id] = name
                    
                    # Update history
                    if track_id not in self.track_name_history:
                        self.track_name_history[track_id] = []
                    self.track_name_history[track_id].append({
                        'name': name,
                        'timestamp': self.frame_count,
                        'bbox': detections[detection_idx]
                    })
        
        # Create new trackers for unmatched detections
        for detection_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(detections[detection_idx])
            self.trackers.append(new_tracker)
            
            # Set initial name if provided
            if detection_names and detection_idx < len(detection_names):
                name = detection_names[detection_idx]
                if name not in ("Unknown", "Unregistered Face", None):
                    self.track_id_to_name[new_tracker.id] = name
        
        # Remove trackers that have been lost
        deleted_trackers = []
        for idx, tracker in enumerate(self.trackers):
            if tracker.time_since_update > self.max_age:
                deleted_trackers.append(idx)
            
            # Remove tracker if it was never confirmed
            elif tracker.hits < self.min_hits and self.frame_count - tracker.age > 0:
                deleted_trackers.append(idx)
        
        # Delete trackers (reverse order to avoid index issues)
        for idx in sorted(deleted_trackers, reverse=True):
            track_id = self.trackers[idx].id
            # Clean up name mapping
            if track_id in self.track_id_to_name:
                del self.track_id_to_name[track_id]
            del self.trackers[idx]
        
        # Return confirmed tracks
        confirmed_tracks = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits:
                track_id = tracker.id
                bbox = tracker.get_state()
                name = self.track_id_to_name.get(track_id, "Unknown")
                confirmed_tracks.append((track_id, bbox, name))
        
        return confirmed_tracks
    
    def get_track_by_id(self, track_id):
        """
        Get tracker by ID
        
        Args:
            track_id: Track ID
            
        Returns:
            Tracker object or None
        """
        for tracker in self.trackers:
            if tracker.id == track_id:
                return tracker
        return None
    
    def get_name_for_track(self, track_id):
        """
        Get name for a track ID
        
        Args:
            track_id: Track ID
            
        Returns:
            Name or None
        """
        return self.track_id_to_name.get(track_id)
    
    def get_all_tracks(self):
        """
        Get all active tracks
        
        Returns:
            List of (track_id, bbox, name) tuples
        """
        tracks = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits:
                track_id = tracker.id
                bbox = tracker.get_state()
                name = self.track_id_to_name.get(track_id, "Unknown")
                tracks.append((track_id, bbox, name))
        return tracks
    
    def reset(self):
        """Reset the tracker"""
        self.trackers = []
        self.track_id_to_name = {}
        self.track_name_history = {}
        KalmanBoxTracker.count = 0


# ── Integration Helper ──────────────────────────────────────────────────

class FaceTracker:
    """
    High-level face tracker that integrates with the attendance system
    """
    
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        """
        Initialize face tracker
        
        Args:
            max_age: Maximum frames without update before track is deleted
            min_hits: Minimum hits required to confirm a track
            iou_threshold: IOU threshold for matching
        """
        self.tracker = SortTracker(max_age, min_hits, iou_threshold)
        self.track_history = {}
        self.frame_count = 0
        
        # Track stability
        self.stable_frames_required = 3
        self.track_confidence = {}
    
    def process_frame(self, detections, names=None):
        """
        Process a frame with face detections and names
        
        Args:
            detections: List of bounding boxes [x, y, w, h]
            names: List of names for each detection (optional)
            
        Returns:
            List of (track_id, bbox, name) tuples
        """
        self.frame_count += 1
        
        # Update tracker
        confirmed_tracks = self.tracker.update(detections, names)
        
        # Update history
        for track_id, bbox, name in confirmed_tracks:
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append({
                'frame': self.frame_count,
                'bbox': bbox,
                'name': name
            })
            
            # Calculate confidence
            if name not in ("Unknown", "Unregistered Face", None):
                self.track_confidence[track_id] = self.track_confidence.get(track_id, 0) + 1
                if self.track_confidence[track_id] >= self.stable_frames_required:
                    # Stable track confirmed
                    pass
        
        return confirmed_tracks
    
    def get_stable_tracks(self, min_frames=3):
        """
        Get tracks that have been stable for at least min_frames
        
        Args:
            min_frames: Minimum number of frames for stability
            
        Returns:
            List of (track_id, bbox, name) tuples
        """
        stable_tracks = []
        for track_id, history in self.track_history.items():
            if len(history) >= min_frames:
                # Get most recent data
                latest = history[-1]
                name = latest['name']
                bbox = latest['bbox']
                if name not in ("Unknown", "Unregistered Face", None):
                    stable_tracks.append((track_id, bbox, name))
        return stable_tracks
    
    def reset(self):
        """Reset the tracker"""
        self.tracker.reset()
        self.track_history = {}
        self.track_confidence = {}
        self.frame_count = 0


def test_tracker():
    """
    Test the face tracker with synthetic data
    """
    print("Testing Face Tracker...")
    
    tracker = FaceTracker()
    
    # Simulate detections over multiple frames
    import random
    
    for frame in range(100):
        # Simulate 2-3 faces
        num_faces = random.randint(2, 3)
        detections = []
        names = []
        
        for i in range(num_faces):
            # Random position with some movement
            x = 100 + i * 200 + random.randint(-20, 20)
            y = 100 + random.randint(-20, 20)
            w = 100 + random.randint(-10, 10)
            h = 100 + random.randint(-10, 10)
            detections.append([x, y, w, h])
            
            # Simulate occasional name changes
            if random.random() > 0.2:
                names.append(f"Person_{i+1}")
            else:
                names.append("Unknown")
        
        # Process frame
        confirmed = tracker.process_frame(detections, names)
        
        if frame % 10 == 0:
            print(f"Frame {frame}: {len(confirmed)} confirmed tracks")
            for track_id, bbox, name in confirmed:
                print(f"   Track {track_id}: {name} at {bbox[:2]}")
    
    # Get stable tracks
    stable = tracker.get_stable_tracks()
    print(f"\nStable tracks: {len(stable)}")
    for track_id, bbox, name in stable:
        print(f"   Track {track_id}: {name} at {bbox[:2]}")


if __name__ == "__main__":
    test_tracker()
