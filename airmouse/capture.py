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
        # Ask for MJPG before the size. UVC webcams advertise uncompressed YUYV
        # only at frame intervals its USB bandwidth allows, so a 720p request
        # negotiates 10 FPS and 1080p just 5 FPS; MJPG carries both at 30. The
        # order matters (V4L2 renegotiates the frame interval when the pixel
        # format changes) and a camera without MJPG simply keeps its own format.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Condition, not a plain lock: consumers sleep until a frame actually
        # lands instead of polling, which kept up to a whole poll interval of
        # avoidable lag in front of every inference.
        self.lock = threading.Condition()
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
                    self.lock.notify()
        except Exception as exc:
            self.error = str(exc)
        finally:
            # Release a waiting consumer immediately so a dead camera surfaces as
            # an error rather than as one timeout per frame.
            with self.lock:
                self.lock.notify_all()
            self.cap.release()

    def take(self, timeout=None):
        """Return the newest frame, waiting up to timeout seconds for one."""
        with self.lock:
            if self.latest is None and timeout:
                self.lock.wait(timeout)
            frame, self.latest = self.latest, None
        return frame

    def close(self):
        self.stop_event.set()
        with self.lock:
            self.lock.notify_all()
        self.thread.join(timeout=2)
