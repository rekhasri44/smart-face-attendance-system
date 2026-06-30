"""
Cache Management Module for Face Embeddings
Handles versioning, validation, and automatic refresh of embedding cache
"""

import os
import json
import hashlib
from datetime import datetime


class CacheManager:
    """Manages embedding cache with versioning and validation"""
    
    def __init__(self, cache_path="embeddings_cache.npy", metadata_path="cache_metadata.json"):
        self.cache_path = cache_path
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self):
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception:
            pass
    
    def _compute_dataset_hash(self, dataset_dir):
        if not os.path.exists(dataset_dir):
            return ""
        
        image_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            return ""
        
        hash_input = []
        for img_path in sorted(image_files):
            rel_path = os.path.relpath(img_path, dataset_dir)
            mtime = os.path.getmtime(img_path)
            hash_input.append(f"{rel_path}:{mtime}")
        
        hash_string = "|".join(hash_input)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def _compute_file_hash(self, file_path):
        if not os.path.exists(file_path):
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def is_cache_valid(self, dataset_dir):
        if not os.path.exists(self.cache_path):
            return False, "Cache file does not exist"
        
        if not self.metadata:
            return False, "Metadata file does not exist"
        
        current_hash = self._compute_dataset_hash(dataset_dir)
        stored_hash = self.metadata.get('dataset_hash', '')
        
        if not current_hash:
            return False, "No images found in dataset"
        
        if current_hash != stored_hash:
            return False, "Dataset changed (hash mismatch)"
        
        return True, "Cache is valid"
    
    def build_metadata(self, dataset_dir, people, embeddings_count, db_size):
        metadata = {
            'version': '1.0',
            'build_date': datetime.now().isoformat(),
            'dataset_hash': self._compute_dataset_hash(dataset_dir),
            'dataset_dir': dataset_dir,
            'people_count': len(people),
            'people': sorted(people),
            'embeddings_count': embeddings_count,
            'db_size': db_size,
            'model_used': 'SFace',
            'cache_path': self.cache_path,
            'cache_hash': self._compute_file_hash(self.cache_path) if os.path.exists(self.cache_path) else ''
        }
        return metadata
    
    def save_metadata(self, metadata):
        self.metadata = metadata
        self._save_metadata()
    
    def get_cache_info(self):
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
        info = self.get_cache_info()
        
        print("\n" + "=" * 60)
        print("📊 CACHE INFORMATION")
        print("=" * 60)
        
        if not info['cache_exists']:
            print("❌ Cache file does not exist")
            return
        
        print(f"📁 Cache Path: {self.cache_path}")
        print(f"📅 Build Date: {info.get('build_date', 'Unknown')}")
        print(f"👥 People Count: {info.get('people_count', 0)}")
        print(f"📊 Embeddings: {info.get('embeddings_count', 0)}")
        print(f"💾 Size: {info.get('cache_size_mb', 0)} MB")
        print(f"🔑 Dataset Hash: {info.get('dataset_hash', 'Unknown')}")
    
    def clear_cache(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            print(f"✅ Removed: {self.cache_path}")
        
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
            print(f"✅ Removed: {self.metadata_path}")
        
        self.metadata = {}
        print("✅ Cache cleared")
