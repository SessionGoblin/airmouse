import threading
import time
import cv2
from .capture import Camera
from .tracking import HandTracker
from .gestures import Features
from . import perf, roles

# A frame older than this is not acted on. Inference is skipped for frames that
# are already this stale when picked up, because the result would be discarded
# after the ~12 ms model call anyway; skipping lets the loop recover on a
# fresher frame instead of compounding the lag.
STALE = .3

# Landmark inference internally works from a small square crop, so pixels beyond
# this width buy no accuracy and cost real time: at 3840x2160 a full-frame detect
# runs ~16 ms against ~12 ms for the same frame scaled to this cap. Downscaling
# first also shrinks the mirror, and hands the UI a frame small enough that the
# preview no longer resizes a 4K image on the main thread. Captures at or below
# the cap are untouched, so the common 640x480 path is unchanged.
WORK_WIDTH = 1280


def analyze(stamp, frame, tracker, now, assigner=None):
    """Mirror the selected frame and, unless it is already stale, run landmark
    inference on it. Returns (frame, hands, start_age, inference_seconds), with
    every hand carrying its features and its assigned role. Kept module-level
    and side-effect free so the freshness gate is testable without a camera or
    the model."""
    # Mirror only the frame actually selected for use, never in the capture
    # thread where most frames are dropped before they are read. Scale down
    # first when oversized, so the mirror runs on the smaller image too.
    height, width = frame.shape[:2]
    if width > WORK_WIDTH:
        frame = cv2.resize(frame, (WORK_WIDTH, max(1, round(height*WORK_WIDTH/width))),
                           interpolation=cv2.INTER_AREA)
    frame = cv2.flip(frame, 1)
    start_age = now - stamp
    if start_age > STALE:
        return frame, [], start_age, 0.0
    start = time.perf_counter()
    hands = tracker.detect(frame)
    inference = time.perf_counter() - start
    aspect = frame.shape[1] / frame.shape[0]
    for hand in hands:
        hand.features = Features.from_landmarks(hand.points, aspect) if hand.points else None
    if assigner is not None:
        assigner.assign(hands, now)
    elif hands:
        hands[0].role = roles.POINTER
    return frame, hands, start_age, inference


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
            settings = self.controller.settings
            tracker = HandTracker(hands=2 if settings.two_hands else 1)
            assigner = roles.RoleAssigner(pointer_side=settings.pointer_side)
            camera = Camera(settings.camera, settings.camera_width, settings.camera_height,
                            hold_fps=settings.hold_fps)
            previous = time.monotonic()
            while not self.stop_event.is_set():
                if camera.error: raise RuntimeError(camera.error)
                item = camera.take(.05)
                if item is None:
                    continue
                stamp, frame = item
                captured = (frame.shape[1], frame.shape[0])
                frame, hands, start_age, inference = analyze(
                    stamp, frame, tracker, time.monotonic(), assigner)
                now = time.monotonic()
                pointer = roles.by_role(hands, roles.POINTER)
                modifier = roles.by_role(hands, roles.MODIFIER)
                features = pointer.features if pointer else None
                gate = modifier.features if modifier else None
                if now-stamp > STALE: features = gate = None
                self.controller.aspect = frame.shape[1] / frame.shape[0]
                state = self.controller.process(features, now, gate)
                fps = 1/max(.001, now-previous)
                previous = now
                self.last_frame = now
                with self.lock:
                    self.latest = (frame, hands, features, state, fps, captured)
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
