"""Static hand-pose templates: normalize, record, match.

Pure landmark math and a JSON template store; no desktop input and no Qt, so
recognition is testable without a camera. A pose is compared as a single frame's
hand shape, independent of where the hand sits in the frame and how far it is
from the camera.
"""
from dataclasses import dataclass, asdict, field
import json
import math
from pathlib import Path

GESTURE_PATH = Path.home() / '.config' / 'airmouse' / 'gestures.json'

WRIST, MIDDLE_MCP = 0, 9
INDEX_MCP, PINKY_MCP = 5, 17
LANDMARKS = 21

# Recorded spread widens a template's accept radius, but it is only a floor
# adjustment: the spread across one held recording measures landmark jitter over
# a second, not how differently the same pose gets formed on a later occasion,
# which is far larger. The floor is what actually decides most matches.
#
# Measured in these units: the same pose re-formed lands .03-.23 away, while
# genuinely different hand shapes sit .67-1.1 apart. The floor sits above the
# first range and the cap well below the second, so there is room for both.
SPREAD_MARGIN = 3.5
MIN_THRESHOLD = .18
MAX_THRESHOLD = .35


def winding(pose):
    """Signed area of the palm triangle, in the upright frame.

    Flips sign between a left and a right hand, which is what makes the two
    distinguishable without trusting the model's handedness classifier. It also
    flips when a hand turns palm-away, so this is "which way the palm winds",
    not "which arm it is on".
    """
    ix, iy = pose[INDEX_MCP]
    px, py = pose[PINKY_MCP]
    return ix*py - iy*px


# |winding| below this means the palm is near edge-on and the sign is not
# trustworthy. Provisional: real hands measured palm-forward sit far above it,
# but this wants checking against the winding readout in --debug.
AMBIGUOUS = .15


def mirrored(pose):
    """Reflect a pose in the upright frame; the wrist and middle MCP stay put."""
    return tuple((-x, y) for x, y in pose)


def canonical(pose):
    """Mirror a pose so both hands share one template space.

    Without this the same shape made with the other hand lands ~.66 away -- as
    far as a genuinely different gesture -- so a recorded pose simply does not
    work on the other hand. Mirroring is applied to the upright frame, where the
    wrist is the origin and the middle MCP is straight up, so both stay fixed.
    """
    return mirrored(pose) if winding(pose) > 0 else tuple(pose)


def normalize(points, aspect=4/3):
    """Landmarks -> (pose, orientation, chirality), or Nones for a bad hand.

    The wrist is the origin and the wrist -> middle-MCP span is the unit, the
    same span ``Features.from_landmarks`` divides pinch distances by, so poses
    and pinch ratios share one unit system. The hand is then rotated upright and
    mirrored into one chirality, so a pose recorded with one hand matches the
    other; the pre-mirror sign is returned so a template can still demand one.

    Rotation is normalized because a pose held at a natural 20-30 degree tilt
    otherwise lands nowhere near its own template, which would force recording
    one template per tilt. The pre-rotation angle is returned separately so a
    template can still require an orientation: thumbs-up and thumbs-down are the
    same shape and differ only by this value.
    """
    if not points or len(points) < LANDMARKS:
        return None, None, None
    ox, oy = points[WRIST][0]*aspect, points[WRIST][1]
    mx, my = points[MIDDLE_MCP][0]*aspect - ox, points[MIDDLE_MCP][1] - oy
    scale = math.hypot(mx, my)
    if scale < 1e-6:
        return None, None, None
    orientation = math.atan2(my, mx)
    # Rotate the wrist -> middle-MCP vector onto -y, which is "up" in image
    # coordinates where y grows downward.
    angle = -orientation - math.pi/2
    cos, sin = math.cos(angle), math.sin(angle)
    pose = []
    for p in points[:LANDMARKS]:
        x, y = p[0]*aspect - ox, p[1] - oy
        pose.append(((x*cos - y*sin)/scale, (x*sin + y*cos)/scale))
    pose = tuple(pose)
    turn = winding(pose)
    # None rather than a coin flip when the palm is edge-on: reporting a
    # confident sign there is what makes a chirality-locked template flicker.
    chirality = None if abs(turn) < AMBIGUOUS else (-1 if turn > 0 else 1)
    return canonical(pose), orientation, chirality


def distance(a, b):
    """Mean per-landmark separation between two normalized poses."""
    return sum(math.dist(p, q) for p, q in zip(a, b)) / len(a)


def angle_delta(a, b):
    """Signed smallest angle from b to a, in radians."""
    return (a - b + math.pi) % (2*math.pi) - math.pi


@dataclass
class Template:
    name: str
    action: str = ''
    argument: str = ''
    pose: tuple = ()
    threshold: float = .18
    # None matches at any tilt; a value requires the hand near that orientation.
    orientation: float = None
    tolerance: float = math.radians(50)
    # None matches either hand; a value requires the palm to wind the same way
    # it did when recorded.
    chirality: int = None
    # Which hand this pose is read from. Chirality stays agnostic even for a
    # modifier pose: the role already selects the hand by position, so locking
    # the mirror too would only break things when the hands are raised in the
    # other order.
    role: str = 'pointer'
    # Only match while the modifier hand holds this named pose. Empty matches
    # in every mode.
    when: str = ''
    # Modifier poses only: holding this one opens stroke drawing.
    gates_drawing: bool = False

    def matches(self, pose, orientation, chirality=None):
        """Distance if this template accepts the pose, else None."""
        if not self.pose or pose is None:
            return None
        gap = distance(pose, self.pose)
        if self.chirality is None:
            # Near edge-on, canonicalization can mirror-flap frame to frame.
            # A template that does not care which hand it is should not inherit
            # that sensitivity, so accept whichever mirror fits better; the cost
            # is one extra 21-point comparison.
            gap = min(gap, distance(mirrored(pose), self.pose))
        elif chirality is not None and chirality != self.chirality:
            return None
        if gap > self.threshold:
            return None
        if self.orientation is not None and orientation is not None:
            if abs(angle_delta(orientation, self.orientation)) > self.tolerance:
                return None
        return gap


def build(name, samples, orientations=None, chiralities=None, **binding):
    """Average recorded samples into a template and size its accept radius.

    The spread across the hold only widens the radius for a genuinely unsteady
    recording; for a normal one it lands under MIN_THRESHOLD and the floor
    decides. That is deliberate -- see the note on the constants above.
    """
    if not samples:
        raise ValueError('No samples recorded')
    mean = tuple(tuple(sum(s[i][k] for s in samples)/len(samples) for k in (0, 1))
                 for i in range(LANDMARKS))
    spread = max((distance(s, mean) for s in samples), default=0.0)
    threshold = min(MAX_THRESHOLD, max(MIN_THRESHOLD, spread*SPREAD_MARGIN))
    orientation = None
    if orientations:
        # Average as unit vectors so the wrap at +/-pi does not average to zero.
        x = sum(math.cos(a) for a in orientations)
        y = sum(math.sin(a) for a in orientations)
        if math.hypot(x, y) > 1e-9:
            orientation = math.atan2(y, x)
    chirality = None
    if chiralities:
        # Majority vote: a frame or two of the palm rolling past edge-on should
        # not decide which hand the template belongs to.
        chirality = 1 if sum(1 for c in chiralities if c == 1)*2 >= len(chiralities) else -1
    return Template(name=name, pose=canonical(mean), threshold=threshold,
                    orientation=orientation, chirality=chirality, **binding)


@dataclass
class Library:
    templates: list = field(default_factory=list)

    def match(self, pose, orientation, chirality=None, role='pointer', mode=None):
        """Closest accepting template for one hand, or None.

        A template scoped to the held mode beats an unscoped one at the same
        distance: the more specific binding is the one the user asked for by
        holding the modifier. Unscoped templates still match in every mode, or
        holding a modifier pose would switch off every ordinary gesture.
        """
        best = None
        for template in self.templates:
            if template.role != role:
                continue
            if template.when and template.when != mode:
                continue
            gap = template.matches(pose, orientation, chirality)
            if gap is None:
                continue
            rank = (0 if template.when else 1, gap)
            if best is None or rank < best[1]:
                best = (template, rank)
        return best[0] if best else None

    def gates(self):
        """Names of modifier poses that open stroke drawing."""
        return {t.name for t in self.templates
                if t.role == 'modifier' and t.gates_drawing}

    def conflict(self, candidate):
        """Existing template a candidate overlaps, if any.

        Overlap is measured against the templates' own radii rather than a fixed
        constant: if the centres are closer than one accept radius, each pose
        falls inside the other's region and the nearer one wins every frame,
        leaving the other unreachable. Checked at record time, because this is
        not a recognition bug that shows up later -- it is a template that can
        never win a comparison.
        """
        for template in self.templates:
            if template.name == candidate.name:
                continue
            if not template.pose or not candidate.pose:
                continue
            # Templates read from different hands, or scoped to different
            # modes, never compete: the same shape can mean one thing on the
            # pointer and another on the modifier.
            if template.role != candidate.role or template.when != candidate.when:
                continue
            if distance(candidate.pose, template.pose) < max(candidate.threshold,
                                                             template.threshold):
                return template
        return None

    def nearest(self, pose, orientation=None, role='pointer'):
        """Closest template and its distance, ignoring thresholds.

        For diagnostics: shows how near a pose came when nothing matched.
        """
        if pose is None or not self.templates:
            return None, None
        scored = [(t, min(distance(pose, t.pose), distance(mirrored(pose), t.pose)))
                  for t in self.templates if t.pose and t.role == role]
        return min(scored, key=lambda pair: pair[1]) if scored else (None, None)

    def replace(self, template):
        self.templates = [t for t in self.templates if t.name != template.name]
        self.templates.append(template)

    def remove(self, name):
        self.templates = [t for t in self.templates if t.name != name]

    @classmethod
    def load(cls, path=None):
        path = Path(path or GESTURE_PATH)
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls()
        templates = []
        for entry in raw if isinstance(raw, list) else []:
            try:
                template = Template(**entry)
                template.pose = tuple(tuple(float(v) for v in p) for p in template.pose)
                # Widen templates recorded under an older, far too tight floor;
                # they would otherwise never match the pose they came from.
                template.threshold = max(template.threshold, MIN_THRESHOLD)
                # Templates recorded before chirality was normalized are stored
                # in whichever mirror the recording hand happened to produce.
                if len(template.pose) == LANDMARKS:
                    template.pose = canonical(template.pose)
                if len(template.pose) == LANDMARKS:
                    templates.append(template)
            except (TypeError, ValueError):
                continue
        return cls(templates)

    def save(self, path=None):
        path = Path(path or GESTURE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps([asdict(t) for t in self.templates], indent=2))
        temporary.replace(path)


class PoseLatch:
    """Confirm a held pose, fire once, and re-arm only after it is released.

    The pointer hand gets this behaviour inside the gesture machine, where it is
    interleaved with the pinch priority chain. The modifier hand has no such
    chain -- it only ever holds a pose -- so it uses this directly rather than
    running a second gesture machine that would try to click with the off hand.
    """

    def __init__(self):
        self.template = None       # what is currently held and confirmed
        self.candidate = None
        self.since = 0.0
        self.latched = False

    def reset(self):
        self.__init__()

    def update(self, template, now, dwell):
        """Feed this frame's match. Returns a template only on the firing frame.

        The held template is exposed as ``self.template`` for as long as it is
        confirmed, which is what makes a modifier pose a mode rather than an
        event.
        """
        if template is None:
            self.template = self.candidate = None
            self.latched = False
            return None
        if self.candidate is None or self.candidate.name != template.name:
            self.candidate, self.since, self.latched = template, now, False
            self.template = None
            return None
        if now - self.since < dwell:
            return None
        self.template = template
        if self.latched:
            return None
        self.latched = True
        return template
