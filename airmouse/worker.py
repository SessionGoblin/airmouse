import threading
import time
import cv2
from .capture import Camera
from .tracking import HandTracker
from .gestures import Features
from . import perf

# A frame older than this is not acted on. Inference is skipped for frames that
# are already this stale when picked up, because the result would be discarded
# after the ~12 ms model call anyway; skipping lets the loop recover on a
# fresher frame instead of compounding the lag.
STALE = .3


def analyze(stamp, frame, tracker, now):
    """Mirror the selected frame and, unless it is already stale, run landmark
    inference on it. Returns (frame, points, confidence, features, start_age,
    inference_seconds). Kept module-level and side-effect free so the freshness
    gate is testable without a camera or the model."""
    # Mirror only the frame actually selected for use, never in the capture
    # thread where most frames are dropped before they are read.
    frame = cv2.flip(frame, 1)
    start_age = now - stamp
    if start_age > STALE:
        return frame, None, None, None, start_age, 0.0
    start = time.perf_counter()
    points, confidence = tracker.detect(frame)
    inference = time.perf_counter() - start
    features = Features.from_landmarks(points, frame.shape[1] / frame.shape[0]) if points else None
    return frame, points, confidence, features, start_age, inference


class VisionWorker:
    def __init__(self, controller):
        self.controller = controller
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.last_frame = time.monotonic()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        camera = tracker = None
        try:
            tracker = HandTracker()
            settings = self.controller.settings
            camera = Camera(settings.camera, settings.camera_width, settings.camera_height)
            previous = time.monotonic()
            while not self.stop_event.is_set():
                if camera.error: raise RuntimeError(camera.error)
                item = camera.take()
                if item is None:
                    self.stop_event.wait(.005)
                    continue
                stamp, frame = item
                frame, points, confidence, features, start_age, inference = analyze(
                    stamp, frame, tracker, time.monotonic())
                now = time.monotonic()
                if now-stamp > STALE: features = None
                state = self.controller.process(features, now)
                fps = 1/max(.001, now-previous)
                previous = now
                self.last_frame = now
                with self.lock:
                    self.latest = (frame, points, confidence, features, state, fps)
                if perf.PROFILER.enabled:
                    perf.PROFILER.record('infer_start_age', start_age*1000)
                    if inference: perf.PROFILER.record('inference', inference*1000)
                    perf.PROFILER.record('output_age', (time.monotonic()-stamp)*1000)
                    perf.PROFILER.tick(stamp)
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                self.controller.pause()
            finally:
                try:
                    if camera: camera.close()
                finally:
                    if tracker: tracker.close()

    def take(self):
        with self.lock:
            result, self.latest = self.latest, None
        return result

    def close(self):
        self.controller.pause()
        self.stop_event.set()
        self.thread.join(timeout=4)
        return not self.thread.is_alive()
