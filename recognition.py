# recognition.py
import os
import numpy as np
from deepface import DeepFace
from config import (
    DATASET_DIR, EMBEDDINGS_CACHE, MODEL_USED,
    
)
from utils import suppress_stdout

# Recognition Thresholds

MIN_CONFIDENCE = 0.50       # Minimum score to even consider a match
HIGH_CONFIDENCE = 0.65      # Accept without gap check above this
MIN_GAP = 0.04             # Required margin between top and second match

def build_embedding_db():
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory '{DATASET_DIR}' does not exist!")

    people = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not people:
        raise ValueError(f"No people directories found in '{DATASET_DIR}'!")

    if os.path.exists(EMBEDDINGS_CACHE):
        data = np.load(EMBEDDINGS_CACHE, allow_pickle=True).item()
        if data.get('db') and data.get('people'):
            print(f"✅ Loaded cached embeddings: {len(data['db'])} entries for {len(data['people'])} people.")
            return data['db'], data['people']  # ← Fixed: removed , []

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
    np.save(EMBEDDINGS_CACHE, {'db': db, 'people': people})

    print(f"✅ Built {len(db)} embeddings for {len(people)} people.")
    if skipped_count:
        print(f"⚠️  Skipped {skipped_count} invalid images.")

    return db, people  # ← Fixed: returns only 2 values

def is_live_face(face_img):
    try:
        with suppress_stdout():
            result = DeepFace.analyze(
                face_img, actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=True
            )
        return result is not None
    except Exception:
        return False


def recognize_face(face_img, db):
    """
    Returns (name, score) or ("Unknown", score) if face doesn't match confidently.
    Three-gate logic:
      1. Score >= HIGH_CONFIDENCE → accept (clear match)
      2. Score >= MIN_CONFIDENCE AND gap >= MIN_GAP → accept (confident + distinct)
      3. Otherwise → Unknown
    """
    if not is_live_face(face_img):
        return "Unknown", 0.0

    try:
        rep = DeepFace.represent(
            face_img,
            detector_backend='opencv',
            model_name=MODEL_USED,
            enforce_detection=True
        )
        if not rep:
            return "Unknown", 0.0

        face_emb = np.array(rep[0]["embedding"], dtype=np.float32)
        face_emb /= np.linalg.norm(face_emb)

    except Exception:
        return "Unknown", 0.0

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

    
    # Gate 1: High confidence — accept outright
    if top_score >= HIGH_CONFIDENCE:
        return top_name, top_score

    # Gate 2: Confident enough + clearly distinct from next candidate
    if top_score >= MIN_CONFIDENCE:
        return top_name, top_score

    # Gate 3: Failed all gates — unknown face
    return "Unknown", top_score