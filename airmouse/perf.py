"""Optional low-overhead latency instrumentation for the vision pipeline.

Disabled unless ``AIRMOUSE_PROFILE=1`` is set (or ``PROFILER.enable()`` is
called). When disabled, ``record``/``tick`` return after a single attribute
read, so the hot path pays almost nothing. Durations are measured with
``time.perf_counter``; frame ages are computed against the ``time.monotonic``
capture timestamp that stamps each frame (the same clock the freshness gate and
the UI use), so ages and gate decisions stay consistent.
"""
import os
import sys
import time
import threading


class Profiler:
    def __init__(self, enabled=None):
        self.enabled = (os.environ.get('AIRMOUSE_PROFILE') == '1') if enabled is None else enabled
        self.interval = 2.0
        self._lock = threading.Lock()
        self._stats = {}          # name -> [count, total_ms, max_ms]
        self._frames = 0
        self._last_ts = 0.0
        self._last_report = time.perf_counter()

    def enable(self):
        self.enabled = True

    def record(self, name, ms):
        if not self.enabled:
            return
        with self._lock:
            s = self._stats.get(name)
            if s is None:
                self._stats[name] = [1, ms, ms]
            else:
                s[0] += 1
                s[1] += ms
                if ms > s[2]:
                    s[2] = ms

    def tick(self, capture_stamp=None):
        """Count one processed frame; emit a periodic stderr summary."""
        if not self.enabled:
            return
        now = time.perf_counter()
        line = None
        with self._lock:
            self._frames += 1
            if capture_stamp is not None:
                self._last_ts = capture_stamp
            span = now - self._last_report
            if span >= self.interval and self._frames:
                parts = [f'{self._frames / span:4.1f} fps', f'cap_ts {self._last_ts:.3f}']
                for name in ('infer_start_age', 'inference', 'output_age', 'preview'):
                    s = self._stats.get(name)
                    if s:
                        parts.append(f'{name} {s[1] / s[0]:.1f}/{s[2]:.1f}ms')
                line = 'airmouse.perf (avg/max) | ' + ' | '.join(parts)
                self._stats.clear()
                self._frames = 0
                self._last_report = now
        if line:
            print(line, file=sys.stderr)


PROFILER = Profiler()
