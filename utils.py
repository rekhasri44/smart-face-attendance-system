# utils.py
import sys
import cv2
import numpy as np
from contextlib import contextmanager
from collections import deque
from typing import Optional


# ── Temporal Consistency / Anti-Flicker ─────────────────────────────────────
class FaceStabilizer:
    """
    Lightweight per-face-slot stabilizer.
    Buffers recent recognition results and only confirms an identity
    once it dominates the rolling window.
    """
    WINDOW = 5          # how many recent frames to consider
    CONFIRM_VOTES = 3   # votes needed to confirm an identity
    STICKY_FRAMES = 3   # consecutive mismatches needed to unseat confirmed id

    def __init__(self):
        self._buffer: deque = deque(maxlen=self.WINDOW)
        self.confirmed: Optional[str] = None   # currently shown label
        self._mismatch_count: int = 0

    def update(self, raw_label: Optional[str]) -> str:
        """
        Feed raw recognition result; returns stable display label.
        raw_label: name string, "Unregistered Face", or None (no face)
        """
        label = raw_label if raw_label is not None else "Unregistered Face"
        self._buffer.append(label)

        # count votes in current window
        vote_counts: dict = {}
        for v in self._buffer:
            vote_counts[v] = vote_counts.get(v, 0) + 1

        top_label = max(vote_counts, key=vote_counts.__getitem__)
        top_votes = vote_counts[top_label]

        if top_votes >= self.CONFIRM_VOTES:
            if self.confirmed is None:
                # first confirmation
                self.confirmed = top_label
                self._mismatch_count = 0
            elif top_label == self.confirmed:
                self._mismatch_count = 0          # still same — reset counter
            else:
                self._mismatch_count += 1
                if self._mismatch_count >= self.STICKY_FRAMES:
                    self.confirmed = top_label    # switch only after enough mismatches
                    self._mismatch_count = 0
        # else: buffer not decisive yet — keep confirmed as-is

        return self.confirmed if self.confirmed is not None else "Detecting..."

    def reset(self):
        self._buffer.clear()
        self.confirmed = None
        self._mismatch_count = 0
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def suppress_stdout():
    class DummyFile:
        def write(self, x): pass
        def flush(self): pass
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stdout = save_stdout


def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


def draw_label(frame, text, x, y, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
    cv2.putText(frame, text, (x, y - 5), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def safe_to_csv(df, filename):
    while True:
        try:
            df.to_csv(filename, index=False)
            print(f"✅ Saved: {filename}")
            break
        except PermissionError:
            input(f"❌ Close {filename} and press Enter...")