from pathlib import Path
import contextlib
import os
import sys
import time
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .roles import Hand

MODEL = Path.home() / '.cache' / 'airmouse' / 'hand_landmarker.task'


@contextlib.contextmanager
def _muted_native_stderr():
    """Silence MediaPipe/TFLite one-time C++ init chatter (absl logs before
    InitGoogle, so GLOG_minloglevel/TF_CPP_MIN_LOG_LEVEL don't apply). Only the
    OS stderr fd is redirected, and only around setup; Python-level stderr and
    all runtime inference errors are untouched."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

def download_model():
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL.with_suffix('.download')
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open('wb') as out:
            while chunk := response.read(1024*1024):
                out.write(chunk)
        if temporary.stat().st_size < 1000000:
            raise RuntimeError('Incomplete model download')
        temporary.replace(MODEL)
    finally:
        temporary.unlink(missing_ok=True)
    return MODEL

class HandTracker:
    def __init__(self, model=MODEL, hands=1):
        if not Path(model).is_file():
            raise RuntimeError('Model missing. Run: python -m airmouse --download-model')
        with _muted_native_stderr():
            self.detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.VIDEO, num_hands=hands,
                min_hand_detection_confidence=.65, min_hand_presence_confidence=.65,
                min_tracking_confidence=.65))
            # One warm-up inference makes the graph's one-time warnings fire (and
            # be swallowed) here, and primes the XNNPACK delegate so the first
            # real frame is not slower than the rest.
            self.detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8)), 0)
        self.timestamp = 0

    def detect(self, frame):
        """Every hand in the frame, in the model's own arbitrary order.

        The order is not stable between frames and the handedness label flickers
        when hands overlap, so neither is used to decide roles; see roles.py.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.timestamp = max(self.timestamp+1, int(time.monotonic()*1000))
        result = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.timestamp)
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[index] if index < len(result.handedness) else None
            # MediaPipe exposes handedness confidence, not per-frame tracking confidence.
            category = handedness[0] if handedness else None
            hands.append(Hand(points=[(p.x, p.y, p.z) for p in landmarks],
                              label=category.category_name if category else '',
                              score=category.score if category else 0.0))
        return hands

    def close(self):
        self.detector.close()
