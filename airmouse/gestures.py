"""Pure landmark interpretation and state machine; no desktop input calls."""
from dataclasses import dataclass
from enum import Enum
import math

from . import poses
from .poses import MIDDLE_MCP

SCROLL_ENGAGE = .15   # seconds holding the pose before scrolling starts
SCROLL_GAIN = 50      # wheel steps per unit of normalized vertical hand travel
SCROLL_STEP_CAP = 6   # max wheel steps emitted per frame

# Drag persistence. Entering a drag needs strong evidence of a pinch, but
# staying in one must not: landmarks go uncertain exactly when a hand is doing
# something -- turning, occluding its own thumb, leaving the frame edge -- and
# reading that as letting go drops the window mid-move. So an established drag
# exits only on a pinch that has clearly opened, and survives frames it cannot
# read at all until the hand has been unreadable for longer than a hiccup.
DRAG_RELEASE_SCALE = 1.35   # how much wider than `release` an established drag's exit is
DRAG_RELEASE_CAP = .70      # ... but never so wide that an open hand cannot release
DRAG_GRACE = .15            # seconds of unusable tracking a held drag survives
DEGRADED_CONFIDENCE = .35   # below this a held drag is running on its trajectory

# Closing a pinch moves the index fingertip, which is also the pointing
# landmark, so the last stretch of a click is hand geometry rather than hand
# movement -- measured at about 30 px of mapped travel on a 1920 desktop. A
# drag must not begin until that has stopped, or the click commits somewhere
# the user was not aiming and the window jumps as it starts.
#
# "Stopped" is a rate test, because the two are far apart: a pinch closes at
# roughly 3.5 units/s while a settled one only wanders at the noise rate, well
# under 1. The timeout is the fallback for a pinch closed so gently that the
# rate never separates.
PINCH_SETTLE_RATE = 1.0     # units of pinch distance per second
PINCH_SETTLE_MAX = .12      # seconds after the press, whatever the rate says

# Why a drag ended. The distinction is the point: only an opened pinch is the
# user letting go, and only that may start a throw.
RELEASE_INTENTIONAL = 'intentional_release'
RELEASE_TIMEOUT = 'tracking_timeout'
RELEASE_PAUSED = 'paused'
RELEASE_DISABLED = 'gesture_disabled'
RELEASE_PREEMPTED = 'preempted'

TRACKING_OK, TRACKING_DEGRADED, TRACKING_LOST = 'ok', 'degraded', 'lost'


def drag_release(settings):
    """Pinch distance at which an established drag counts as opened.

    Wider than `release`, which is the hysteresis: entering a drag still needs
    `pinch`, so a press is as deliberate as it ever was, but staying in one
    needs only that the fingers have not visibly parted. Capped, so a
    permissive `release` cannot widen the exit past where a hand is plainly open
    and leave a drag impossible to end.
    """
    return min(max(settings.release, settings.release*DRAG_RELEASE_SCALE), DRAG_RELEASE_CAP)


def drag_grace(settings):
    return max(0.0, getattr(settings, 'drag_grace', DRAG_GRACE))


def pinch_confidence(left, settings):
    """How firmly a drag pose is held: 1 at the press threshold, 0 at the exit.

    Not a detector confidence -- MediaPipe does not expose one per frame -- but
    the same normalized pinch distance the thresholds use, read as a continuous
    quantity instead of a boolean. That is what lets the drag filter lean on its
    trajectory as a pinch loosens rather than waiting for it to cross a line.
    """
    if left is None or not math.isfinite(left):
        return 0.0
    ceiling = drag_release(settings)
    if left <= settings.pinch:
        return 1.0
    if left >= ceiling:
        return 0.0
    return (ceiling-left)/max(1e-6, ceiling-settings.pinch)


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
    THROWN = 'throwing'
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
    # The middle knuckle: where the hand *is*, as opposed to where it is
    # pointing. Closing a pinch swings the index fingertip by some 30 px of
    # mapped travel while this barely moves, so it is the honest answer to
    # "has the hand gone anywhere" -- which is the question the drag deadzone
    # is asking. Pointing still uses the fingertip. Defaulted so the built-in
    # gestures can still be exercised with a bare four-field Features.
    palm: tuple = None

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
                   pose, orientation, chirality, points[MIDDLE_MCP][:2])

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
        # Last frame that carried usable features, which is what the drag grace
        # period is measured from: the first unreadable frame is not the moment
        # tracking was lost, it is the moment we noticed.
        self.seen = None
        self.confidence = 0.0
        self.tracking = TRACKING_LOST
        self.release_reason = None
        # Whether the pinch that started this press has stopped closing. Until
        # it has, fingertip movement is the pinch forming, not the hand going
        # anywhere; see PINCH_SETTLE_RATE.
        self.pinch_settled = False
        self.previous_left = None
        self.previous_at = None

    def reset(self, paused=True):
        actions = [('up',)] if self.down else []
        reason = self.release_reason
        self.__init__(self.library)     # templates survive a reset
        # ... and so does why the last drag ended: it is a record of something
        # that already happened, and the throw path and the diagnostics pane
        # both read it after the reset that a lost hand triggers.
        self.release_reason = reason
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
        if held:
            self.release_reason = RELEASE_PREEMPTED
        return [('up',)] if held else []

    def drag_hold(self, now, settings):
        """Whether a held drag survives a frame it cannot read.

        Measured from the last usable frame, not from the first bad one, so a
        stall that hands over a frame late does not buy itself a fresh grace
        period. One authority for the policy, consulted by both this machine
        and the controller, so the two can never disagree about whether a drag
        is still latched.
        """
        if not self.down or self.seen is None:
            return False
        return 0 <= now - self.seen <= drag_grace(settings)

    def latch(self, now, dragging=None):
        """Record one frame of a drag held through tracking it cannot use.

        Called both from here and by the controller, which coasts the drag
        anchor without stepping the machine at all -- so this is the one place
        that says what a latched frame does to the machine's view of itself.
        """
        self.tracking = TRACKING_LOST
        self.confidence = 0.0
        if dragging is None:
            dragging = now-self.pressed_at > .22
        self.state = State.DRAGGING if dragging else State.PINCH_DOWN
        return self.state

    def step(self, f, now, settings, enabled, mode=None):
        if not enabled:
            # An emergency stop is not a gesture: nothing is latched through it.
            actions = self.reset(paused=True)
            if actions:
                self.release_reason = RELEASE_PAUSED
            return actions
        if f is None:
            if self.drag_hold(now, settings):
                # Latched. No move, no release: the controller coasts the drag
                # anchor on its trajectory until the hand reads again.
                self.latch(now)
                return []
            actions = self.reset(paused=False)
            if actions:
                self.release_reason = RELEASE_TIMEOUT
            return actions
        if self.down and not self.pinch_settled:
            rate = (abs(f.left-self.previous_left)/max(1e-3, now-self.previous_at)
                    if self.previous_left is not None and self.previous_at is not None
                    else 0.0)
            if rate < PINCH_SETTLE_RATE or now-self.pressed_at >= PINCH_SETTLE_MAX:
                self.pinch_settled = True
        self.previous_left, self.previous_at = f.left, now
        self.seen = now
        self.confidence = pinch_confidence(f.left, settings)
        self.tracking = (TRACKING_OK if not self.down or self.confidence >= DEGRADED_CONFIDENCE
                         else TRACKING_DEGRADED)
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
            # Hysteresis: entering needed `pinch`, staying needs only that the
            # pinch has not clearly opened. Anything between the two thresholds
            # is a degraded pose, not a release, and keeps the drag latched.
            if not settings.left or f.left >= drag_release(settings):
                self.down = False
                self.last_action = now
                self.state = State.RELEASE
                self.candidate = None
                self.release_reason = (RELEASE_DISABLED if not settings.left
                                       else RELEASE_INTENTIONAL)
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
        template = self.library.match(f.pose, f.orientation, f.chirality,
                                      role='pointer', mode=mode) if (
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
                self.release_reason = None
                self.pinch_settled = False
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
