import os
import sys
import time
import numpy as np
import cv2
import pytest

from airmouse import capture as C
from airmouse.config import Settings


class FakeCap:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.sets = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def read(self):
        time.sleep(0.05)
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fake_camera(monkeypatch):
    pins = []
    monkeypatch.setattr(cv2, 'VideoCapture', FakeCap)
    monkeypatch.setattr(C, 'pin_dynamic_framerate', lambda index: pins.append(index) or 1)
    restored = []
    monkeypatch.setattr(C, 'restore_dynamic_framerate', lambda i, p: restored.append((i, p)))
    cam = C.Camera(0, 1280, 720)
    try:
        yield cam, pins, restored
    finally:
        cam.close()


def test_linux_opens_v4l2_and_asks_for_mjpg_before_size(fake_camera):
    cam, pins, restored = fake_camera
    assert cam.cap.args[0] == 0
    if sys.platform == 'linux':
        assert cam.cap.args[1] == cv2.CAP_V4L2
    fourcc = [i for i, (p, _) in enumerate(cam.cap.sets) if p == cv2.CAP_PROP_FOURCC]
    width = [i for i, (p, _) in enumerate(cam.cap.sets) if p == cv2.CAP_PROP_FRAME_WIDTH]
    assert fourcc and width and fourcc[0] < width[0]
    assert cam.cap.sets[fourcc[0]][1] == cv2.VideoWriter_fourcc(*'MJPG')
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in cam.cap.sets
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 720) in cam.cap.sets
    assert pins == [0]
    assert [p for p, _ in cam.cap.sets].count(cv2.CAP_PROP_FPS) == 2
    assert all(v == 30 for p, v in cam.cap.sets if p == cv2.CAP_PROP_FPS)
    cam.close()
    assert (0, 1) in restored


def test_hold_fps_off_does_not_touch_exposure(monkeypatch):
    pins, restored = [], []
    monkeypatch.setattr(cv2, 'VideoCapture', FakeCap)
    monkeypatch.setattr(C, 'pin_dynamic_framerate', lambda index: pins.append(index) or 1)
    monkeypatch.setattr(C, 'restore_dynamic_framerate', lambda i, p: restored.append((i, p)))
    cam = C.Camera(0, hold_fps=False)
    try:
        assert pins == []
        assert [p for p, _ in cam.cap.sets].count(cv2.CAP_PROP_FPS) == 1
    finally:
        cam.close()
    assert restored == [] or all(p is None for _, p in restored)


def test_windows_uses_directshow(monkeypatch):
    monkeypatch.setattr(C.sys, 'platform', 'win32')
    monkeypatch.setattr(cv2, 'VideoCapture', FakeCap)
    cam = C.Camera(0, hold_fps=False)
    try:
        assert cam.cap.args == (0, cv2.CAP_DSHOW)
    finally:
        cam.close()


def test_pin_dynamic_framerate_is_noop_off_linux(monkeypatch):
    monkeypatch.setattr(C.sys, 'platform', 'darwin')
    opened = []
    monkeypatch.setattr(C.os, 'open', lambda *a, **k: opened.append(a) or 3)
    assert C.pin_dynamic_framerate(0) is None
    assert opened == []


@pytest.mark.skipif(sys.platform != 'linux', reason='Requires Linux fcntl')
def test_pin_dynamic_framerate_clears_only_when_on(monkeypatch):
    monkeypatch.setattr(C.sys, 'platform', 'linux')
    current = {C.V4L2_CID_EXPOSURE_AUTO_PRIORITY: 1}
    written = []
    monkeypatch.setattr(os, 'open', lambda *a, **k: 7)
    monkeypatch.setattr(os, 'close', lambda fd: None)

    def ioctl(fd, req, buf, mutate=True):
        nr = req & 0xff
        if nr == 27:
            buf.value = current[buf.id]
        else:
            current[buf.id] = buf.value
            written.append((buf.id, buf.value))
        return 0

    import fcntl
    monkeypatch.setattr(fcntl, 'ioctl', ioctl)
    previous = C.pin_dynamic_framerate(2)
    assert previous == 1
    assert written == [(C.V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)]
    C.restore_dynamic_framerate(2, previous)
    assert current[C.V4L2_CID_EXPOSURE_AUTO_PRIORITY] == 1


@pytest.mark.skipif(sys.platform != 'linux', reason='Requires Linux fcntl')
def test_pin_dynamic_framerate_skips_when_already_off(monkeypatch):
    monkeypatch.setattr(C.sys, 'platform', 'linux')
    monkeypatch.setattr(os, 'open', lambda *a, **k: 7)
    monkeypatch.setattr(os, 'close', lambda fd: None)
    sets = []

    def ioctl(fd, req, buf, mutate=True):
        nr = req & 0xff
        if nr == 27:
            buf.value = 0
        else:
            sets.append(buf.value)
        return 0

    import fcntl
    monkeypatch.setattr(fcntl, 'ioctl', ioctl)
    assert C.pin_dynamic_framerate(0) is None
    assert sets == []


@pytest.mark.skipif(sys.platform != 'linux', reason='Requires Linux fcntl')
def test_pin_dynamic_framerate_ignores_missing_device(monkeypatch):
    monkeypatch.setattr(C.sys, 'platform', 'linux')
    monkeypatch.setattr(os, 'open', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError('nope')))
    assert C.pin_dynamic_framerate(9) is None


def test_hold_fps_setting_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr('airmouse.config.CONFIG_PATH', tmp_path / 'settings.json')
    Settings(hold_fps=False).save()
    assert Settings.load().hold_fps is False
    with pytest.raises(ValueError):
        Settings(hold_fps='yes').validate()
