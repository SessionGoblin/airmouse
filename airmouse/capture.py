import threading
import time
import cv2

class Camera:
    """Single latest-frame slot prevents inference from building a video backlog."""
    def __init__(self, index, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f'Cannot open webcam {index}. Check camera permissions or other apps.')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    self.error = 'Webcam stopped returning frames'
                    break
                stamp = time.monotonic()
                # Store the raw frame and let the consumer mirror only the frame
                # it actually uses; flipping here would burn CPU (and hold the
                # lock) on frames that are overwritten before they are ever read.
                # cap.read() returns a fresh array each call, so the stored
                # reference is not aliased by the next capture.
                with self.lock:
                    self.latest = (stamp, frame)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.cap.release()

    def take(self):
        with self.lock:
            frame, self.latest = self.latest, None
        return frame

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
