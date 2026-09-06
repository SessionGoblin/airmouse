import threading
import time
from .capture import Camera
from .tracking import HandTracker
from .gestures import Features

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
            camera = Camera(self.controller.settings.camera)
            previous = time.monotonic()
            while not self.stop_event.is_set():
                if camera.error: raise RuntimeError(camera.error)
                item = camera.take()
                if item is None:
                    self.stop_event.wait(.005)
                    continue
                stamp, frame = item
                points, confidence = tracker.detect(frame)
                now = time.monotonic()
                features = Features.from_landmarks(points, frame.shape[1]/frame.shape[0]) if points else None
                if now-stamp > .3: features = None
                state = self.controller.process(features, now)
                fps = 1/max(.001, now-previous)
                previous = now
                self.last_frame = now
                with self.lock:
                    self.latest = (frame, points, confidence, features, state, fps)
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
