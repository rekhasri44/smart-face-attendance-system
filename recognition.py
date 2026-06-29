import os
import numpy as np
from deepface import DeepFace
from config import (
    DATASET_DIR, EMBEDDINGS_CACHE, MODEL_USED,
    MIN_CONFIDENCE_FOR_REVIEW, HIGH_CONFIDENCE
)
from utils import suppress_stdout
from cache_manager import CacheManager

# Recognition Thresholds
MIN_CONFIDENCE = 0.50
HIGH_CONFIDENCE = 0.65
MIN_GAP = 0.04


def build_embedding_db():
    """
    Build embedding database with cache versioning
    """
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory '{DATASET_DIR}' does not exist!")

    people = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not people:
        raise ValueError(f"No people directories found in '{DATASET_DIR}'!")

    # ── Cache Versioning ──────────────────────────────────────────────────
    cache_manager = CacheManager(
        cache_path=EMBEDDINGS_CACHE,
        metadata_path="cache_metadata.json"
    )
    
    # Check if cache is valid
    is_valid, reason = cache_manager.is_cache_valid(DATASET_DIR)
    
    if is_valid:
        print(f"✅ Cache is valid: {reason}")
        data = np.load(EMBEDDINGS_CACHE, allow_pickle=True).item()
        if data.get('db') and data.get('people'):
            print(f"✅ Loaded cached embeddings: {len(data['db'])} entries for {len(data['people'])} people.")
            cache_manager.display_cache_info()
            return data['db'], data['people']
    else:
        print(f"🔄 Cache invalid: {reason}")
        print("🔄 Rebuilding embeddings...")
    # ────────────────────────────────────────────────────────────────────

    print("📂 Building embeddings from dataset...")

    db = []
    valid_people = set()
    skipped_count = 0

    for person in people:
        person_dir = os.path.join(DATASET_DIR, person)
        img_files = [f for f in os.listdir(person_dir) if not f.startswith('.')]

        for img_name in img_files:
            img_path = os.path.join(person_dir, img_name)
            try:
                with suppress_stdout():
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        detector_backend='opencv',
                        model_name=MODEL_USED,
                        enforce_detection=True
                    )[0]["embedding"]
                embedding = np.array(embedding, dtype=np.float32)
                embedding /= np.linalg.norm(embedding)
                db.append({"name": person, "embedding": embedding})
                valid_people.add(person)
            except Exception:
                skipped_count += 1

    people = sorted(list(valid_people))
    
    # ── Save cache with metadata ─────────────────────────────────────────
    # Save the embeddings
    np.save(EMBEDDINGS_CACHE, {'db': db, 'people': people})
    
    # Build and save metadata
    cache_size = os.path.getsize(EMBEDDINGS_CACHE) if os.path.exists(EMBEDDINGS_CACHE) else 0
    metadata = cache_manager.build_metadata(
        dataset_dir=DATASET_DIR,
        people=people,
        embeddings_count=len(db),
        db_size=cache_size
    )
    cache_manager.save_metadata(metadata)
    
    print(f"✅ Built {len(db)} embeddings for {len(people)} people.")
    print(f"✅ Saved cache metadata to cache_metadata.json")
    if skipped_count:
        print(f"⚠️  Skipped {skipped_count} invalid images.")
    
    cache_manager.display_cache_info()
    # ────────────────────────────────────────────────────────────────────

    return db, people


def is_live_face(face_img):
    try:
        with suppress_stdout():
            result = DeepFace.analyze(
                face_img, 
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=True
            )
        return result is not None
    except Exception:
        return False


def recognize_face(face_img, db):
    """
    Returns (name, score, review_needed)
    review_needed: True if the match is in the borderline zone
    """
    if not is_live_face(face_img):
        return "Unknown", 0.0, False

    try:
        rep = DeepFace.represent(
            face_img,
            detector_backend='opencv',
            model_name=MODEL_USED,
            enforce_detection=True
        )
        if not rep:
            return "Unknown", 0.0, False

        face_emb = np.array(rep[0]["embedding"], dtype=np.float32)
        face_emb /= np.linalg.norm(face_emb)

    except Exception:
        return "Unknown", 0.0, False

    top_name, top_score, second_score = None, -1.0, -1.0

    for entry in db:
        db_emb = entry["embedding"] / np.linalg.norm(entry["embedding"])
        score = float(np.dot(face_emb, db_emb))

        if score > top_score:
            second_score = top_score
            top_score = score
            top_name = entry["name"]
        elif score > second_score:
            second_score = score

    # Check if review is needed
    review_needed = False
    
    # Gate 1: High confidence — accept outright
    if top_score >= HIGH_CONFIDENCE:
        return top_name, top_score, False
    
    # Gate 2: Borderline zone — needs review
    if top_score >= MIN_CONFIDENCE_FOR_REVIEW and top_score < HIGH_CONFIDENCE:
        review_needed = True
        if (top_score - second_score) >= MIN_GAP:
            return top_name, top_score, True
        else:
            return "Unknown", top_score, True
    
    # Gate 3: Low confidence — reject
    return "Unknown", top_score, False
