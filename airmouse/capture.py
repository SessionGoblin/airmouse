import threading
import time
import cv2

class Camera:
    """Single latest-frame slot prevents inference from building a video backlog."""
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f'Cannot open webcam {index}. Check camera permissions or other apps.')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
                with self.lock:
                    self.latest = (time.monotonic(), cv2.flip(frame, 1))
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
