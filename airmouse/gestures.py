"""Pure landmark interpretation and state machine; no desktop input calls."""
from dataclasses import dataclass
from enum import Enum
import math

class State(str, Enum):
    IDLE = 'idle'
    POINTING = 'pointing'
    PINCH_DOWN = 'pinch-down'
    DRAGGING = 'dragging'
    RELEASE = 'release'
    RIGHT_CLICK = 'right-click'
    SCROLLING = 'scrolling'
    PAUSED = 'paused'

@dataclass
class Features:
    point: tuple
    left: float
    right: float
    scroll: bool

    @classmethod
    def from_landmarks(cls, points, aspect=4/3):
        def distance(a, b):
            return math.hypot((points[a][0]-points[b][0])*aspect, points[a][1]-points[b][1])
        scale = max(distance(0, 9), .02)
        extended = {tip: distance(0, tip) > distance(0, tip-2)*1.18 for tip in (8, 12, 16, 20)}
        plane = [(p[0]*aspect,p[1],0) for p in points]
        angles = {tip: joint_angle(plane,tip-3,tip-2,tip) for tip in (8,12,16,20)}
        return cls(points[8][:2], distance(4, 8)/scale, distance(4, 12)/scale,
                   extended[8] and extended[12] and angles[8] > 155 and angles[12] > 155
                   and angles[16] < 135 and angles[20] < 135
                   and not extended[16] and not extended[20])

class GestureMachine:
    def __init__(self):
        self.state = State.PAUSED
        self.down = False
        self.armed = False
        self.present_since = None
        self.candidate = None
        self.since = 0
        self.last_action = -100
        self.right_latched = False
        self.scroll_y = None
        self.pressed_at = 0
        self.scroll_exit = None

    def reset(self, paused=True):
        actions = [('up',)] if self.down else []
        self.__init__()
        self.state = State.PAUSED if paused else State.IDLE
        return actions

    def step(self, f, now, settings, enabled):
        if not enabled or f is None:
            return self.reset(paused=not enabled)
        if self.present_since is None:
            self.present_since = now
        if not self.armed:
            self.state = State.IDLE
            if f.left > settings.release and f.right > settings.release and not f.scroll:
                if now-self.present_since >= settings.dwell:
                    self.armed = True
            else:
                self.present_since = now
            return []
        if f.right > settings.release:
            self.right_latched = False
        elif self.right_latched and settings.right:
            # Holding a right pinch must not move the newly opened menu.
            self.state = State.RIGHT_CLICK
            return []
        if self.down:
            if f.left >= settings.release or not settings.left:
                self.down = False
                self.last_action = now
                self.state = State.RELEASE
                self.candidate = None
                return [('up',)]
            self.state = State.DRAGGING if now-self.pressed_at > .22 else State.PINCH_DOWN
            return [('move', f.point)]
        if self.state == State.SCROLLING and settings.scroll:
            if not f.scroll:
                if self.scroll_exit is None:
                    self.scroll_exit = now
                if now-self.scroll_exit < .12:
                    return []
            else:
                self.scroll_exit = None
        desired = ('left' if settings.left and f.left < settings.pinch else
                   'right' if settings.right and f.right < settings.pinch and not self.right_latched else
                   'scroll' if settings.scroll and f.scroll and f.left > settings.release and f.right > settings.release else None)
        if desired != self.candidate:
            self.candidate, self.since = desired, now
            self.scroll_y = None
            self.scroll_exit = None
        if desired:
            # Freeze cursor while confirming an intentional gesture.
            if now-self.since < (settings.dwell if desired == 'scroll' else settings.debounce):
                return []
            if desired == 'scroll':
                self.state = State.SCROLLING
                if self.scroll_y is None:
                    self.scroll_y = f.point[1]
                delta = int((self.scroll_y-f.point[1])*35)
                if delta:
                    self.scroll_y -= delta/35
                    return [('scroll', max(-5, min(5, delta)))]
                return []
            if now-self.last_action < settings.cooldown:
                return []
            self.last_action = now
            if desired == 'left':
                self.down, self.pressed_at = True, now
                self.state = State.PINCH_DOWN
                return [('down',)]
            self.right_latched = True
            self.state = State.RIGHT_CLICK
            return [('right',)]
        self.state = State.POINTING
        return [('move', f.point)]


def joint_angle(points, a, b, c):
    """3D landmark angle for tuning; gesture decisions use normalized distances."""
    u = tuple(x-y for x,y in zip(points[a],points[b]))
    v = tuple(x-y for x,y in zip(points[c],points[b]))
    length = math.sqrt(sum(x*x for x in u)*sum(x*x for x in v))
    return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(u,v))/max(length,1e-9)))))
