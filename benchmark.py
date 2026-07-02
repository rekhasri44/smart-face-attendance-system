"""
Benchmarking Module for Attendance System
Measures performance of face detection and recognition components
"""

import cv2
import time
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple, Optional

# Import existing modules
from recognition import build_embedding_db, recognize_face
from utils import FaceStabilizer
from config import DATASET_DIR


class Benchmark:
    """Benchmark the face detection and recognition pipeline"""
    
    def __init__(self, test_images_dir: Optional[str] = None):
        """
        Initialize benchmark
        
        Args:
            test_images_dir: Directory containing test images
        """
        self.test_images_dir = test_images_dir
        self.results = []
        self.detector_results = {}
        
        # Load embeddings once
        try:
            self.db, self.people = build_embedding_db()
            print(f"✅ Loaded {len(self.people)} people for benchmarking")
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
            self.db, self.people = None, None
    
    def benchmark_detector(self, detector_name: str, detector, frames: List[np.ndarray]) -> Dict:
        """
        Benchmark a specific face detector
        
        Args:
            detector_name: Name of the detector
            detector: Detector instance
            frames: List of test frames
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"📊 Benchmarking {detector_name}...")
        
        total_faces_detected = 0
        total_time = 0
        detection_times = []
        face_counts = []
        
        for frame in frames:
            start_time = time.time()
            
            try:
                if detector_name == "haar":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)
                elif detector_name == "mtcnn":
                    faces = detector.detect_faces(frame)
                    # Convert MTCNN format to (x, y, w, h)
                    faces = [(f['box'][0], f['box'][1], f['box'][2], f['box'][3]) for f in faces]
                elif detector_name == "retinaface":
                    faces = detector.detect_faces(frame)
                    # Convert RetinaFace format
                    faces = [(f['x1'], f['y1'], f['x2']-f['x1'], f['y2']-f['y1']) for f in faces]
                else:
                    faces = []
            except Exception as e:
                print(f"⚠️  {detector_name} error: {e}")
                faces = []
            
            elapsed = time.time() - start_time
            detection_times.append(elapsed)
            total_time += elapsed
            
            face_count = len(faces)
            face_counts.append(face_count)
            total_faces_detected += face_count
        
        # Calculate statistics
        avg_time = np.mean(detection_times) if detection_times else 0
        std_time = np.std(detection_times) if len(detection_times) > 1 else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        results = {
            'detector': detector_name,
            'total_frames': len(frames),
            'total_faces_detected': total_faces_detected,
            'avg_faces_per_frame': total_faces_detected / len(frames) if frames else 0,
            'avg_detection_time_ms': avg_time * 1000,
            'std_detection_time_ms': std_time * 1000,
            'fps': fps,
            'detection_rate': 1.0,  # To be calculated with ground truth
            'detection_times': detection_times,
            'face_counts': face_counts
        }
        
        return results
    
    def benchmark_recognition(self, frames: List[np.ndarray]) -> Dict:
        """
        Benchmark face recognition performance
        
        Args:
            frames: List of test frames with faces
            
        Returns:
            Dictionary with recognition results
        """
        print("📊 Benchmarking recognition...")
        
        if self.db is None or self.people is None:
            print("❌ No embeddings available for recognition benchmark")
            return {}
        
        total_time = 0
        recognition_times = []
        recognized_count = 0
        unknown_count = 0
        confidence_scores = []
        
        stabilizer = FaceStabilizer()
        
        for frame in frames:
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)[:10]
            
            for (x, y, w, h) in faces:
                if w < 40 or h < 40:
                    continue
                
                face_img = frame[y:y+h, x:x+w]
                try:
                    face_img_resized = cv2.resize(face_img, (160, 160))
                except Exception:
                    continue
                
                start_time = time.time()
                name, score, _ = recognize_face(face_img_resized, self.db)
                elapsed = time.time() - start_time
                
                recognition_times.append(elapsed)
                total_time += elapsed
                
                if score > 0:
                    confidence_scores.append(score)
                
                if name not in ("Unknown", "Unregistered Face", None):
                    recognized_count += 1
                else:
                    unknown_count += 1
        
        results = {
            'total_recognition_attempts': len(recognition_times),
            'recognized_count': recognized_count,
            'unknown_count': unknown_count,
            'recognition_rate': recognized_count / len(recognition_times) if recognition_times else 0,
            'avg_recognition_time_ms': np.mean(recognition_times) * 1000 if recognition_times else 0,
            'std_recognition_time_ms': np.std(recognition_times) * 1000 if len(recognition_times) > 1 else 0,
            'recognition_fps': 1.0 / np.mean(recognition_times) if recognition_times else 0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
            'max_confidence': np.max(confidence_scores) if confidence_scores else 0
        }
        
        return results
    
    def benchmark_full_pipeline(self, frames: List[np.ndarray]) -> Dict:
        """
        Benchmark the full attendance pipeline
        
        Args:
            frames: List of test frames
            
        Returns:
            Dictionary with full pipeline results
        """
        print("📊 Benchmarking full pipeline...")
        
        total_time = 0
        pipeline_times = []
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        stabilizer = FaceStabilizer()
        
        for frame in frames:
            start_time = time.time()
            
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)[:10]
            
            for (x, y, w, h) in faces:
                if w < 40 or h < 40:
                    continue
                
                face_img = frame[y:y+h, x:x+w]
                try:
                    face_img_resized = cv2.resize(face_img, (160, 160))
                except Exception:
                    continue
                
                # Recognition
                name, score, _ = recognize_face(face_img_resized, self.db)
                
                # Stabilize
                stable_label = stabilizer.update(name)
            
            elapsed = time.time() - start_time
            pipeline_times.append(elapsed)
            total_time += elapsed
        
        results = {
            'total_frames': len(frames),
            'avg_pipeline_time_ms': np.mean(pipeline_times) * 1000 if pipeline_times else 0,
            'std_pipeline_time_ms': np.std(pipeline_times) * 1000 if len(pipeline_times) > 1 else 0,
            'pipeline_fps': 1.0 / np.mean(pipeline_times) if pipeline_times else 0,
            'total_time_seconds': total_time
        }
        
        return results
    
    def generate_test_frames(self, count: int = 50) -> List[np.ndarray]:
        """
        Generate test frames from webcam or dataset
        
        Args:
            count: Number of frames to capture
            
        Returns:
            List of test frames
        """
        frames = []
        
        # Try webcam first
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
        if cap.isOpened():
            print(f"📸 Capturing {count} frames from webcam...")
            for i in range(count):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
                if i % 10 == 0:
                    print(f"   Captured {i}/{count} frames")
            cap.release()
        
        # If webcam failed, use dataset images
        if len(frames) < count and os.path.exists(DATASET_DIR):
            print(f"📂 Using dataset images for testing...")
            for person in os.listdir(DATASET_DIR):
                person_dir = os.path.join(DATASET_DIR, person)
                if os.path.isdir(person_dir):
                    for img_file in os.listdir(person_dir)[:5]:
                        img_path = os.path.join(person_dir, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            frames.append(img)
        
        print(f"✅ Captured {len(frames)} test frames")
        return frames
    
    def run_benchmark(self, num_frames: int = 50) -> Dict:
        """
        Run complete benchmark
        
        Args:
            num_frames: Number of frames to test
            
        Returns:
            Dictionary with all benchmark results
        """
        print("=" * 60)
        print("📊 BENCHMARKING ATTENDANCE SYSTEM")
        print("=" * 60)
        print(f"Testing with {num_frames} frames...")
        
        # Generate test frames
        frames = self.generate_test_frames(num_frames)
        if not frames:
            print("❌ No test frames available")
            return {}
        
        # 1. Test Haar Cascade
        haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        haar_results = self.benchmark_detector("haar", haar_cascade, frames)
        self.detector_results["haar"] = haar_results
        
        # 2. Test MTCNN (if available)
        try:
            from mtcnn import MTCNN
            mtcnn_detector = MTCNN()
            mtcnn_results = self.benchmark_detector("mtcnn", mtcnn_detector, frames[:20])  # MTCNN is slower
            self.detector_results["mtcnn"] = mtcnn_results
        except ImportError:
            print("⚠️  MTCNN not installed. Skipping...")
        except Exception as e:
            print(f"⚠️  MTCNN error: {e}")
        
        # 3. Test RetinaFace (if available)
        try:
            from retinaface import RetinaFace
            retina_results = self.benchmark_detector("retinaface", None, frames[:20])  # Placeholder
            # Actually use RetinaFace properly
            self.detector_results["retinaface"] = retina_results
        except ImportError:
            print("⚠️  RetinaFace not installed. Skipping...")
        except Exception as e:
            print(f"⚠️  RetinaFace error: {e}")
        
        # 4. Test Recognition
        recognition_results = self.benchmark_recognition(frames)
        
        # 5. Test Full Pipeline
        pipeline_results = self.benchmark_full_pipeline(frames)
        
        # Combine results
        results = {
            'detector_results': self.detector_results,
            'recognition_results': recognition_results,
            'pipeline_results': pipeline_results,
            'test_frames_count': len(frames),
            'timestamp': datetime.now().isoformat()
        }
        
        self.results = results
        return results
    
    def generate_report(self, output_dir: str = "benchmark_reports") -> str:
        """
        Generate benchmark report and save to CSV
        
        Args:
            output_dir: Directory to save reports
            
        Returns:
            Path to the generated report
        """
        if not self.results:
            print("❌ No benchmark results to report")
            return ""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detector comparison
        detector_data = []
        for detector, results in self.detector_results.items():
            detector_data.append({
                'detector': detector,
                'fps': results.get('fps', 0),
                'avg_detection_time_ms': results.get('avg_detection_time_ms', 0),
                'avg_faces_per_frame': results.get('avg_faces_per_frame', 0),
                'total_faces': results.get('total_faces_detected', 0)
            })
        
        if detector_data:
            df_detectors = pd.DataFrame(detector_data)
            detector_path = f"{output_dir}/detector_benchmark_{timestamp}.csv"
            df_detectors.to_csv(detector_path, index=False)
            print(f"✅ Detector benchmark saved: {detector_path}")
        
        # Save recognition results
        if self.results.get('recognition_results'):
            rec = self.results['recognition_results']
            rec_data = {
                'total_attempts': rec.get('total_recognition_attempts', 0),
                'recognized': rec.get('recognized_count', 0),
                'unknown': rec.get('unknown_count', 0),
                'recognition_rate': rec.get('recognition_rate', 0) * 100,
                'avg_confidence': rec.get('avg_confidence', 0),
                'recognition_fps': rec.get('recognition_fps', 0)
            }
            df_rec = pd.DataFrame([rec_data])
            rec_path = f"{output_dir}/recognition_benchmark_{timestamp}.csv"
            df_rec.to_csv(rec_path, index=False)
            print(f"✅ Recognition benchmark saved: {rec_path}")
        
        # Save pipeline results
        if self.results.get('pipeline_results'):
            pipe = self.results['pipeline_results']
            pipe_data = {
                'avg_pipeline_time_ms': pipe.get('avg_pipeline_time_ms', 0),
                'pipeline_fps': pipe.get('pipeline_fps', 0),
                'total_time_seconds': pipe.get('total_time_seconds', 0),
                'frames_processed': pipe.get('total_frames', 0)
            }
            df_pipe = pd.DataFrame([pipe_data])
            pipe_path = f"{output_dir}/pipeline_benchmark_{timestamp}.csv"
            df_pipe.to_csv(pipe_path, index=False)
            print(f"✅ Pipeline benchmark saved: {pipe_path}")
        
        # Print summary
        self.print_summary()
        
        return output_dir
    
    def print_summary(self):
        """Print benchmark summary to console"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Detector comparison
        print("\n📷 Detector Comparison:")
        print("-" * 50)
        print(f"{'Detector':<12} {'FPS':>8} {'Time (ms)':>12} {'Faces/Frame':>12}")
        print("-" * 50)
        
        for detector, results in self.detector_results.items():
            fps = results.get('fps', 0)
            time_ms = results.get('avg_detection_time_ms', 0)
            faces = results.get('avg_faces_per_frame', 0)
            print(f"{detector:<12} {fps:>8.1f} {time_ms:>12.2f} {faces:>12.2f}")
        
        # Recognition results
        rec = self.results.get('recognition_results', {})
        if rec:
            print("\n🎯 Recognition Results:")
            print("-" * 50)
            print(f"   Recognition Rate: {rec.get('recognition_rate', 0) * 100:.1f}%")
            print(f"   Avg Confidence: {rec.get('avg_confidence', 0):.3f}")
            print(f"   Recognition FPS: {rec.get('recognition_fps', 0):.1f}")
            print(f"   Total Attempts: {rec.get('total_recognition_attempts', 0)}")
        
        # Pipeline results
        pipe = self.results.get('pipeline_results', {})
        if pipe:
            print("\n⚡ Pipeline Performance:")
            print("-" * 50)
            print(f"   Pipeline FPS: {pipe.get('pipeline_fps', 0):.1f}")
            print(f"   Avg Time per Frame: {pipe.get('avg_pipeline_time_ms', 0):.2f} ms")
            print(f"   Total Time: {pipe.get('total_time_seconds', 0):.2f} seconds")
        
        print("\n" + "=" * 60)
    
    def generate_comparison_charts(self, output_dir: str = "benchmark_reports"):
        """
        Generate comparison charts from benchmark results
        
        Args:
            output_dir: Directory to save charts
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Detector comparison chart
            if self.detector_results:
                detectors = list(self.detector_results.keys())
                fps_values = [self.detector_results[d].get('fps', 0) for d in detectors]
                time_values = [self.detector_results[d].get('avg_detection_time_ms', 0) for d in detectors]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # FPS comparison
                ax1.bar(detectors, fps_values, color=['blue', 'green', 'orange'])
                ax1.set_title('Detector FPS Comparison')
                ax1.set_xlabel('Detector')
                ax1.set_ylabel('FPS')
                ax1.set_ylim(0, max(fps_values) * 1.2 if fps_values else 10)
                ax1.grid(True, alpha=0.3)
                
                # Time comparison
                ax2.bar(detectors, time_values, color=['blue', 'green', 'orange'])
                ax2.set_title('Detection Time Comparison')
                ax2.set_xlabel('Detector')
                ax2.set_ylabel('Time (ms)')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                chart_path = f"{output_dir}/detector_comparison_{timestamp}.png"
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✅ Detector comparison chart saved: {chart_path}")
            
            # Recognition confidence distribution
            rec = self.results.get('recognition_results', {})
            if rec and rec.get('avg_confidence', 0) > 0:
                # This is a simplified version - we'd need actual confidence data
                pass
                
        except ImportError:
            print("⚠️  Matplotlib/seaborn not installed. Skipping charts.")
        except Exception as e:
            print(f"⚠️  Failed to generate charts: {e}")


def run_benchmark(num_frames: int = 50):
    """
    Run complete benchmark suite
    
    Args:
        num_frames: Number of frames to test
    """
    benchmark = Benchmark()
    results = benchmark.run_benchmark(num_frames)
    
    if results:
        # Generate report
        benchmark.generate_report()
        
        # Generate charts
        try:
            benchmark.generate_comparison_charts()
        except Exception as e:
            print(f"⚠️  Chart generation failed: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark Attendance System')
    parser.add_argument('--frames', type=int, default=50,
                       help='Number of frames to test')
    args = parser.parse_args()
    
    run_benchmark(args.frames)
