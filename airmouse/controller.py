import threading
from .gestures import GestureMachine
from .mapping import CursorMapper

class Controller:
    """Serialized control gate, including emergency release from keyboard thread."""
    def __init__(self, settings, bounds, backend=None):
        self.settings, self.backend = settings, backend
        self.mapper = CursorMapper(bounds)
        self.machine = GestureMachine()
        self.lock = threading.RLock()
        self.enabled = False
        self.last = None
        self.target = None
        self.last_seen = None

    def pause(self):
        with self.lock:
            self.enabled = False
            try:
                if self.backend: self.backend.up()
            finally:
                self.machine.reset()
                self.mapper.filter.reset()
                self.last = None
                self.target = None

    def resume(self):
        with self.lock:
            if self.backend is None: return False
            self.pause()
            self.enabled = True
            return True

    def process(self, features, now):
        with self.lock:
            if features is None or (self.last is not None and now-self.last > .3):
                if self.backend and self.machine.down: self.backend.up()
                self.machine.reset(paused=not self.enabled)
                self.mapper.filter.reset()
            dt = min(.1, now-self.last) if self.last is not None else 1/30
            self.last = now
            self.last_seen = now if features else None
            if self.mapper.filter.value is None and features:
                # Slew from current desktop position on X11; virtual absolute pointer
                # starts at centre and approaches the hand slowly on acquisition.
                origin = (self.mapper.bounds[0]+self.mapper.bounds[2]/2,
                          self.mapper.bounds[1]+self.mapper.bounds[3]/2)
                if self.backend and hasattr(self.backend, 'mouse'):
                    origin = tuple(self.backend.mouse.position)
                self.mapper.filter.reset(origin)
            actions = self.machine.step(features, now, self.settings, self.enabled)
            for action in actions:
                if not self.backend: continue
                if action[0] == 'move':
                    previous = self.target or self.mapper.filter.value
                    desired = self.mapper.update(action[1], self.settings, dt)
                    # Limit speed during reacquisition and normal operation.
                    cap = max(self.mapper.bounds[2:]) * dt * 1.7
                    from math import dist
                    distance = dist(previous, desired)
                    if distance > cap:
                        desired = tuple(round(a+(b-a)*cap/distance) for a,b in zip(previous,desired))
                    self.target = desired
                    self.mapper.filter.value = desired
                    self.backend.move(*desired)
                elif action[0] == 'scroll': self.backend.scroll(action[1])
                else: getattr(self.backend, action[0])()
            return self.machine.state.value

    def close(self):
        with self.lock:
            try: self.pause()
            finally:
                if self.backend: self.backend.close()
                self.backend = None
