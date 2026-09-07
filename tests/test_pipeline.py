import numpy as np

from airmouse import worker as W
from airmouse import perf
from airmouse.gestures import Features


class FakeTracker:
    """Records every frame handed to inference so tests can assert whether the
    expensive model call ran, and on which (mirrored) pixels."""
    def __init__(self, points=None):
        self.points = points
        self.calls = []

    def detect(self, frame):
        self.calls.append(frame.copy())
        if self.points is None:
            return None, None
        return self.points, ('Right', 0.9)


def test_stale_frame_skips_inference():
    tracker = FakeTracker(points=[(0.5, 0.5, 0.0)] * 21)
    frame = np.zeros((48, 64, 3), np.uint8)
    _, points, conf, feats, start_age, inference = W.analyze(
        100.0, frame, tracker, 100.0 + W.STALE + 0.05)
    assert tracker.calls == []          # the ~12 ms model call was skipped
    assert points is None and conf is None and feats is None
    assert inference == 0.0
    assert start_age > W.STALE


def test_fresh_frame_runs_inference_on_mirrored_frame():
    points = [(i / 21.0, i / 21.0, 0.0) for i in range(21)]
    tracker = FakeTracker(points=points)
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[:, :10] = 255                 # bright stripe on the left edge
    _, got, conf, feats, start_age, inference = W.analyze(10.0, frame, tracker, 10.01)
    assert len(tracker.calls) == 1
    assert got is points
    assert isinstance(feats, Features)
    seen = tracker.calls[0]
    # The mirror moved the bright stripe from the left edge to the right edge.
    assert seen[:, -10:].mean() > seen[:, :10].mean()


def test_fresh_frame_without_hand_still_runs_inference():
    tracker = FakeTracker(points=None)
    _, points, conf, feats, start_age, inference = W.analyze(
        0.0, np.zeros((48, 64, 3), np.uint8), tracker, 0.01)
    assert len(tracker.calls) == 1      # a fresh empty frame is still inferred
    assert points is None and feats is None


def test_profiler_disabled_is_noop():
    p = perf.Profiler(enabled=False)
    p.record('inference', 12.0)
    p.tick(1.0)
    assert p._stats == {}


def test_profiler_accumulates_when_enabled():
    p = perf.Profiler(enabled=True)
    p.record('inference', 2.0)
    p.record('inference', 6.0)
    count, total, mx = p._stats['inference']
    assert count == 2 and total == 8.0 and mx == 6.0


def test_oversized_frame_is_scaled_before_inference():
    """A capture wider than WORK_WIDTH reaches the model downscaled: the extra
    pixels do not improve landmarks and cost ~4 ms per frame at 4K."""
    tracker = FakeTracker(points=[(0.5, 0.5, 0.0)] * 21)
    frame = np.zeros((2160, 3840, 3), np.uint8)
    out, *_ = W.analyze(0.0, frame, tracker, 0.01)
    assert tracker.calls[0].shape[:2] == (W.WORK_WIDTH * 2160 // 3840, W.WORK_WIDTH)
    assert out.shape == tracker.calls[0].shape


def test_frame_within_cap_is_not_resized():
    tracker = FakeTracker(points=None)
    frame = np.zeros((480, 640, 3), np.uint8)
    out, *_ = W.analyze(0.0, frame, tracker, 0.01)
    assert out.shape == (480, 640, 3)
    assert tracker.calls[0].shape == (480, 640, 3)
