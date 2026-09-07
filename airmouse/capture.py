import os
import sys
import threading
import time
import cv2

# UVC "Exposure, Dynamic Framerate" (linux/v4l2-controls.h). Logitech RightLight
# uses this to drop below the negotiated interval when the shutter would exceed it.
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903


def _v4l2_control(index, cid, value=None):
    """Get a V4L2 control, or set it when value is given. None if unavailable."""
    if sys.platform != 'linux':
        return None
    import ctypes
    import fcntl

    class v4l2_control(ctypes.Structure):
        _fields_ = [('id', ctypes.c_uint32), ('value', ctypes.c_int32)]

    size = ctypes.sizeof(v4l2_control)
    # _IOWR('V', nr, struct v4l2_control) — G_CTRL is 27, S_CTRL is 28
    nr = 28 if value is not None else 27
    request = (3 << 30) | (ord('V') << 8) | nr | (size << 16)
    control = v4l2_control(cid, 0 if value is None else int(value))
    try:
        fd = os.open(f'/dev/video{index}', os.O_RDWR)
    except OSError:
        return None
    try:
        fcntl.ioctl(fd, request, control)
        return control.value
    except OSError:
        return None
    finally:
        os.close(fd)


def pin_dynamic_framerate(index):
    """If the camera is allowed to drop FPS for brightness, turn that off.

    Returns the previous control value so the caller can restore it, or None
    when the control is missing, already off, or not on Linux. Auto exposure
    and gain are left alone; the picture may go darker rather than stuttering.
    """
    previous = _v4l2_control(index, V4L2_CID_EXPOSURE_AUTO_PRIORITY)
    if not previous:
        return None
    _v4l2_control(index, V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)
    return previous


def restore_dynamic_framerate(index, previous):
    if previous is None:
        return
    _v4l2_control(index, V4L2_CID_EXPOSURE_AUTO_PRIORITY, previous)


class Camera:
    """Single latest-frame slot prevents inference from building a video backlog."""
    def __init__(self, index, width=640, height=480, hold_fps=True):
        backend = cv2.CAP_V4L2 if sys.platform == 'linux' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
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
        # Format changes can reset UVC controls, so apply after the mode is set
        # and re-assert 30 FPS in case the camera had already dropped to 15.
        self._device = index
        self._fps_priority = pin_dynamic_framerate(index) if hold_fps else None
        if self._fps_priority is not None:
            self.cap.set(cv2.CAP_PROP_FPS, 30)
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
            restore_dynamic_framerate(self._device, self._fps_priority)
            self._fps_priority = None

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
        # If the capture thread did not finish, still put the camera back.
        restore_dynamic_framerate(self._device, self._fps_priority)
        self._fps_priority = None
