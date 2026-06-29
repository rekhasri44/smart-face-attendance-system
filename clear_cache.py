#!/usr/bin/env python
"""
Utility script to clear the embedding cache
"""

import os
from cache_manager import CacheManager
from config import EMBEDDINGS_CACHE, CACHE_METADATA


def clear_cache():
    """Clear the embedding cache"""
    print("=" * 60)
    print("🗑️  CACHE CLEAR TOOL")
    print("=" * 60)
    
    manager = CacheManager(
        cache_path=EMBEDDINGS_CACHE,
        metadata_path=CACHE_METADATA
    )
    
    # Show current cache info
    manager.display_cache_info()
    
    # Confirm
    print("\n⚠️  This will delete the cache files.")
    response = input("Are you sure? (y/n): ")
    
    if response.lower() == 'y':
        manager.clear_cache()
        print("✅ Cache cleared successfully!")
    else:
        print("❌ Operation cancelled.")


if __name__ == "__main__":
    clear_cache()
