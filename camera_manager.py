"""
Camera Manager Module
Handles multiple camera streams with frame skipping and switching
"""

import cv2
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np


class CameraManager:
    """Manages multiple camera streams with frame skipping"""
    
    def __init__(self, camera_config: Dict, active_cameras: list, frame_skip: int = 2):
        """
        Initialize camera manager
        
        Args:
            camera_config: Dictionary of camera configurations
            active_cameras: List of camera keys to use
            frame_skip: Process every Nth frame
        """
        self.camera_config = camera_config
        self.active_cameras = active_cameras
        self.frame_skip = frame_skip
        
        self.cameras = {}
        self.frame_counters = {}
        self.last_frames = {}
        self.camera_stats = {}
        
        # Initialize cameras
        self._initialize_cameras()
        
        # Current active camera index
        self.current_camera_idx = 0
        self.last_switch_time = time.time()
        self.switch_interval = 5  # seconds
    
    def _initialize_cameras(self):
        """Initialize all active cameras"""
        for cam_key in self.active_cameras:
            if cam_key in self.camera_config:
                config = self.camera_config[cam_key]
                index = config.get('index', 0)
                
                try:
                    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
                    if cap.isOpened():
                        self.cameras[cam_key] = cap
                        self.frame_counters[cam_key] = 0
                        self.camera_stats[cam_key] = {
                            'fps': 0,
                            'frames_processed': 0,
                            'last_frame_time': datetime.now()
                        }
                        print(f"✅ Camera '{cam_key}' (index {index}) initialized")
                    else:
                        print(f"❌ Failed to open camera '{cam_key}' (index {index})")
                except Exception as e:
                    print(f"❌ Error initializing camera '{cam_key}': {e}")
    
    def get_next_frame(self) -> Tuple[Optional[np.ndarray], Optional[str], Optional[Dict]]:
        """
        Get next frame from active cameras with round-robin switching
        
        Returns:
            (frame, camera_key, camera_stats)
        """
        if not self.cameras:
            return None, None, None
        
        # Get current camera key
        cam_keys = list(self.cameras.keys())
        if self.current_camera_idx >= len(cam_keys):
            self.current_camera_idx = 0
        
        cam_key = cam_keys[self.current_camera_idx]
        cap = self.cameras[cam_key]
        
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            # Try to reconnect
            print(f"⚠️  Camera '{cam_key}' lost connection, reconnecting...")
            self._reconnect_camera(cam_key)
            return None, None, None
        
        # Update frame counter
        self.frame_counters[cam_key] += 1
        
        # Skip frames for processing (but still read to keep buffer fresh)
        if self.frame_counters[cam_key] % self.frame_skip != 0:
            return None, cam_key, self.camera_stats.get(cam_key)
        
        # Update stats
        stats = self.camera_stats.get(cam_key, {})
        stats['frames_processed'] = stats.get('frames_processed', 0) + 1
        stats['last_frame_time'] = datetime.now()
        
        # Calculate FPS
        if 'last_fps_time' in stats:
            elapsed = (datetime.now() - stats['last_fps_time']).total_seconds()
            if elapsed > 0:
                stats['fps'] = 1.0 / elapsed
        stats['last_fps_time'] = datetime.now()
        
        return frame, cam_key, stats
    
    def switch_camera(self):
        """Switch to the next camera"""
        cam_keys = list(self.cameras.keys())
        if not cam_keys:
            return
        
        self.current_camera_idx = (self.current_camera_idx + 1) % len(cam_keys)
        self.last_switch_time = time.time()
        
        cam_key = cam_keys[self.current_camera_idx]
        print(f"🔄 Switched to camera: {cam_key}")
        
        return cam_key
    
    def _reconnect_camera(self, cam_key: str):
        """Attempt to reconnect a camera"""
        if cam_key in self.cameras:
            self.cameras[cam_key].release()
            del self.cameras[cam_key]
        
        if cam_key in self.camera_config:
            index = self.camera_config[cam_key].get('index', 0)
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
                if cap.isOpened():
                    self.cameras[cam_key] = cap
                    self.frame_counters[cam_key] = 0
                    print(f"✅ Camera '{cam_key}' reconnected")
                else:
                    print(f"❌ Failed to reconnect camera '{cam_key}'")
            except Exception as e:
                print(f"❌ Error reconnecting camera '{cam_key}': {e}")
    
    def release_all(self):
        """Release all camera resources"""
        for cam_key, cap in self.cameras.items():
            try:
                cap.release()
                print(f"✅ Released camera: {cam_key}")
            except Exception:
                pass
        self.cameras.clear()
    
    def get_camera_info(self) -> Dict:
        """Get information about all cameras"""
        info = {}
        for cam_key in self.cameras:
            info[cam_key] = {
                'config': self.camera_config.get(cam_key, {}),
                'stats': self.camera_stats.get(cam_key, {}),
                'frames_processed': self.frame_counters.get(cam_key, 0),
                'is_active': cam_key in self.cameras
            }
        return info
    
    def display_camera_info(self):
        """Display camera information in console"""
        print("\n" + "=" * 50)
        print("📷 CAMERA STATUS")
        print("=" * 50)
        
        for cam_key, info in self.get_camera_info().items():
            status = "✅ Active" if info['is_active'] else "❌ Inactive"
            fps = info.get('stats', {}).get('fps', 0)
            frames = info.get('frames_processed', 0)
            print(f"{cam_key}: {status} | FPS: {fps:.1f} | Frames: {frames}")
        
        cam_keys = list(self.cameras.keys())
        if cam_keys:
            current = cam_keys[self.current_camera_idx % len(cam_keys)]
            print(f"\n🔄 Current Camera: {current}")
        print("=" * 50)


def test_cameras():
    """Test function for camera manager"""
    from config import CAMERA_CONFIG, ACTIVE_CAMERAS, CAMERA_FRAME_SKIP
    
    manager = CameraManager(
        camera_config=CAMERA_CONFIG,
        active_cameras=ACTIVE_CAMERAS,
        frame_skip=CAMERA_FRAME_SKIP
    )
    
    print("Press 'q' to quit")
    print("Press 's' to switch camera")
    
    while True:
        frame, cam_key, stats = manager.get_next_frame()
        
        if frame is not None:
            # Add camera info to frame
            cv2.putText(frame, f"Camera: {cam_key}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if stats:
                fps = stats.get('fps', 0)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Multi-Camera Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            manager.switch_camera()
    
    manager.release_all()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_cameras()
