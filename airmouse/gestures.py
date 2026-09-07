"""Pure landmark interpretation and state machine; no desktop input calls."""
from dataclasses import dataclass
from enum import Enum
import math

from . import poses

SCROLL_ENGAGE = .15   # seconds holding the pose before scrolling starts
SCROLL_GAIN = 50      # wheel steps per unit of normalized vertical hand travel
SCROLL_STEP_CAP = 6   # max wheel steps emitted per frame

class State(str, Enum):
    IDLE = 'idle'
    POINTING = 'pointing'
    PINCH_DOWN = 'pinch-down'
    DRAGGING = 'dragging'
    RELEASE = 'release'
    RIGHT_CLICK = 'right-click'
    SCROLLING = 'scrolling'
    CUSTOM = 'custom-pose'
    DRAWING = 'drawing'
    PAUSED = 'paused'

@dataclass
class Features:
    point: tuple
    left: float
    right: float
    scroll: bool
    # Whole-hand shape for custom templates. Defaulted so the built-in gestures
    # can still be exercised with a bare four-field Features.
    pose: tuple = None
    orientation: float = None
    chirality: int = None

    @classmethod
    def from_landmarks(cls, points, aspect=4/3):
        def distance(a, b):
            return math.hypot((points[a][0]-points[b][0])*aspect, points[a][1]-points[b][1])
        scale = max(distance(0, 9), .02)
        extended = {tip: distance(0, tip) > distance(0, tip-2)*1.12 for tip in (8, 12, 16, 20)}
        plane = [(p[0]*aspect,p[1],0) for p in points]
        angles = {tip: joint_angle(plane,tip-3,tip-2,tip) for tip in (8,12,16,20)}
        # Scroll pose recognition is forgiving on how straight the raised fingers
        # are and how tightly the others are curled; the structural guards
        # (index+middle up, ring+little folded) still keep pointing and an open
        # palm from being read as a scroll.
        pose, orientation, chirality = poses.normalize(points, aspect)
        return cls(points[8][:2], distance(4, 8)/scale, distance(4, 12)/scale,
                   extended[8] and extended[12] and angles[8] > 150 and angles[12] > 150
                   and angles[16] < 145 and angles[20] < 145
                   and not extended[16] and not extended[20],
                   pose, orientation, chirality)

class GestureMachine:
    def __init__(self, library=None):
        self.library = library
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
        self.custom_latched = False
        self.custom_template = None

    def reset(self, paused=True):
        actions = [('up',)] if self.down else []
        self.__init__(self.library)     # templates survive a reset
        self.state = State.PAUSED if paused else State.IDLE
        return actions

    def release(self):
        """Drop a held button without losing the armed state.

        Used when another mode takes over mid-drag: a full reset would force
        the user to re-arm with a neutral hand every time.
        """
        held = self.down
        self.down = False
        self.candidate = None
        self.state = State.IDLE
        return [('up',)] if held else []

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
        # Custom poses are tested before the pinches. A deliberate whole-hand
        # shape is the stronger signal, and the pinch thresholds are permissive
        # enough that shapes like a closed fist read as a click. The recorder
        # warns when a template shadows a built-in, so anything that got saved
        # was accepted knowing that.
        template = self.library.match(f.pose, f.orientation, f.chirality) if (
            self.library is not None and settings.custom) else None
        if template is None:
            self.custom_latched = False
        elif self.custom_latched:
            # Hold the pose without re-firing, and do not move the cursor.
            self.state = State.CUSTOM
            return []
        self.custom_template = template
        desired = ('custom:'+template.name if template is not None else
                   'left' if settings.left and f.left < settings.pinch else
                   'right' if settings.right and f.right < settings.pinch and not self.right_latched else
                   'scroll' if settings.scroll and f.scroll and f.left > settings.release and f.right > settings.release else None)
        if desired != self.candidate:
            self.candidate, self.since = desired, now
            self.scroll_y = None
            self.scroll_exit = None
        if desired:
            # Freeze cursor while confirming an intentional gesture.
            hold = (SCROLL_ENGAGE if desired == 'scroll' else
                    settings.custom_dwell if desired.startswith('custom:') else
                    settings.debounce)
            if now-self.since < hold:
                return []
            if desired == 'scroll':
                self.state = State.SCROLLING
                if self.scroll_y is None:
                    self.scroll_y = f.point[1]
                # Clamp first, then advance the anchor only by what we emit, so
                # fast strokes carry their capped remainder into later frames
                # instead of being silently dropped.
                delta = max(-SCROLL_STEP_CAP, min(SCROLL_STEP_CAP, int((self.scroll_y-f.point[1])*SCROLL_GAIN)))
                if delta:
                    self.scroll_y -= delta/SCROLL_GAIN
                    return [('scroll', delta)]
                return []
            if now-self.last_action < settings.cooldown:
                return []
            self.last_action = now
            if desired.startswith('custom:'):
                # Latch so holding the pose fires once rather than every frame.
                self.custom_latched = True
                self.state = State.CUSTOM
                return [('custom', self.custom_template)]
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
