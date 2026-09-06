from pathlib import Path
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL = Path.home() / '.cache' / 'airmouse' / 'hand_landmarker.task'
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
    def __init__(self, model=MODEL):
        if not Path(model).is_file():
            raise RuntimeError('Model missing. Run: python -m airmouse --download-model')
        self.detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=.65, min_hand_presence_confidence=.65,
            min_tracking_confidence=.65))
        self.timestamp = 0

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.timestamp = max(self.timestamp+1, int(time.monotonic()*1000))
        result = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.timestamp)
        if not result.hand_landmarks:
            return None, None
        points = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
        # MediaPipe exposes handedness confidence, not per-frame tracking confidence.
        category = result.handedness[0][0]
        return points, (category.category_name, category.score)

    def close(self):
        self.detector.close()
