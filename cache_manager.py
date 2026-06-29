"""
Cache Management Module for Face Embeddings
Handles versioning, validation, and automatic refresh of embedding cache
"""

import os
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path


class CacheManager:
    """
    Manages embedding cache with versioning and validation
    """
    
    def __init__(self, cache_path: str = "embeddings_cache.npy", 
                 metadata_path: str = "cache_metadata.json"):
        """
        Initialize cache manager
        
        Args:
            cache_path: Path to the cache file (.npy)
            metadata_path: Path to metadata file (.json)
        """
        self.cache_path = cache_path
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from file"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to file"""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save metadata: {e}")
    
    def _compute_dataset_hash(self, dataset_dir: str) -> str:
        """
        Compute a hash of the dataset directory
        
        Args:
            dataset_dir: Path to dataset directory
            
        Returns:
            SHA-256 hash of all image files
        """
        if not os.path.exists(dataset_dir):
            return ""
        
        # Collect all image files
        image_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            return ""
        
        # Create hash based on file paths and modification times
        hash_input = []
        for img_path in sorted(image_files):
            # Get relative path from dataset directory
            rel_path = os.path.relpath(img_path, dataset_dir)
            # Get modification time
            mtime = os.path.getmtime(img_path)
            # Add to hash input
            hash_input.append(f"{rel_path}:{mtime}")
        
        # Compute hash
        hash_string = "|".join(hash_input)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file"""
        if not os.path.exists(file_path):
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def is_cache_valid(self, dataset_dir: str) -> Tuple[bool, str]:
        """
        Check if cache is valid for the current dataset
        
        Args:
            dataset_dir: Path to dataset directory
            
        Returns:
            (is_valid, reason)
        """
        # Check if cache exists
        if not os.path.exists(self.cache_path):
            return False, "Cache file does not exist"
        
        # Check if metadata exists
        if not self.metadata:
            return False, "Metadata file does not exist"
        
        # Check dataset hash
        current_hash = self._compute_dataset_hash(dataset_dir)
        stored_hash = self.metadata.get('dataset_hash', '')
        
        if not current_hash:
            return False, "No images found in dataset"
        
        if current_hash != stored_hash:
            return False, f"Dataset changed (hash mismatch)"
        
        # Check number of people
        people_count = self.metadata.get('people_count', 0)
        if people_count == 0:
            return False, "No people in cache metadata"
        
        # Check cache file hash (optional - verify integrity)
        cache_hash = self.metadata.get('cache_hash', '')
        if cache_hash:
            current_cache_hash = self._compute_file_hash(self.cache_path)
            if current_cache_hash != cache_hash:
                return False, "Cache file corrupted"
        
        # Check timestamp (warn if too old)
        build_date = self.metadata.get('build_date', '')
        if build_date:
            try:
                build_time = datetime.fromisoformat(build_date)
                days_old = (datetime.now() - build_time).days
                if days_old > 30:
                    # Not invalid, but we can warn
                    print(f"⚠️  Cache is {days_old} days old. Consider rebuilding.")
            except Exception:
                pass
        
        return True, "Cache is valid"
    
    def build_metadata(self, dataset_dir: str, people: list, 
                       embeddings_count: int, db_size: int) -> Dict:
        """
        Build metadata for the cache
        
        Args:
            dataset_dir: Path to dataset directory
            people: List of person names
            embeddings_count: Number of embeddings built
            db_size: Size of the database in bytes
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'version': '1.0',
            'build_date': datetime.now().isoformat(),
            'dataset_hash': self._compute_dataset_hash(dataset_dir),
            'dataset_dir': dataset_dir,
            'people_count': len(people),
            'people': sorted(people),
            'embeddings_count': embeddings_count,
            'db_size': db_size,
            'model_used': 'SFace',  # From config
            'cache_path': self.cache_path,
            'cache_hash': self._compute_file_hash(self.cache_path) if os.path.exists(self.cache_path) else ''
        }
        return metadata
    
    def save_metadata(self, metadata: Dict):
        """Save metadata to file"""
        self.metadata = metadata
        self._save_metadata()
    
    def get_cache_info(self) -> Dict:
        """Get information about the current cache"""
        info = {
            'cache_exists': os.path.exists(self.cache_path),
            'metadata_exists': os.path.exists(self.metadata_path),
        }
        
        if self.metadata:
            info.update({
                'build_date': self.metadata.get('build_date', 'Unknown'),
                'people_count': self.metadata.get('people_count', 0),
                'people': self.metadata.get('people', []),
                'embeddings_count': self.metadata.get('embeddings_count', 0),
                'dataset_hash': self.metadata.get('dataset_hash', '')[:8] + '...',
                'version': self.metadata.get('version', 'Unknown'),
            })
        
        if os.path.exists(self.cache_path):
            size = os.path.getsize(self.cache_path)
            info['cache_size_mb'] = round(size / (1024 * 1024), 2)
        
        return info
    
    def display_cache_info(self):
        """Display cache information in a readable format"""
        info = self.get_cache_info()
        
        print("\n" + "=" * 60)
        print("📊 CACHE INFORMATION")
        print("=" * 60)
        
        if not info['cache_exists']:
            print("❌ Cache file does not exist")
            return
        
        print(f"📁 Cache Path: {self.cache_path}")
        print(f"📁 Metadata Path: {self.metadata_path}")
        print(f"📅 Build Date: {info.get('build_date', 'Unknown')}")
        print(f"👥 People Count: {info.get('people_count', 0)}")
        print(f"📊 Embeddings: {info.get('embeddings_count', 0)}")
        print(f"💾 Size: {info.get('cache_size_mb', 0)} MB")
        print(f"🔑 Dataset Hash: {info.get('dataset_hash', 'Unknown')}")
        print(f"📌 Version: {info.get('version', 'Unknown')}")
        
        # Show people
        people = info.get('people', [])
        if people:
            print(f"\n👤 People ({len(people)}):")
            for i, person in enumerate(people, 1):
                print(f"   {i}. {person}")
    
    def clear_cache(self):
        """Clear the cache files"""
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            print(f"✅ Removed: {self.cache_path}")
        
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
            print(f"✅ Removed: {self.metadata_path}")
        
        self.metadata = {}
        print("✅ Cache cleared")


def get_cache_manager():
    """Singleton instance of cache manager"""
    return CacheManager()


if __name__ == "__main__":
    # Test the cache manager
    manager = CacheManager()
    manager.display_cache_info()
