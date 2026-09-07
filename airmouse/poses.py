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
LANDMARKS = 21

# Recorded spread is multiplied by this to set a template's accept radius, then
# clamped: too tight and the pose never fires, too loose and it shadows others.
SPREAD_MARGIN = 2.5
MIN_THRESHOLD = .06
MAX_THRESHOLD = .35
# Two templates closer than this are near-duplicates; whichever is nearer wins
# every frame and the other is effectively unreachable.
DISTINCT = .12


def normalize(points, aspect=4/3):
    """Landmarks -> (pose, orientation), or (None, None) for a degenerate hand.

    The wrist is the origin and the wrist -> middle-MCP span is the unit, the
    same span ``Features.from_landmarks`` divides pinch distances by, so poses
    and pinch ratios share one unit system. The hand is then rotated upright.

    Rotation is normalized because a pose held at a natural 20-30 degree tilt
    otherwise lands nowhere near its own template, which would force recording
    one template per tilt. The pre-rotation angle is returned separately so a
    template can still require an orientation: thumbs-up and thumbs-down are the
    same shape and differ only by this value.
    """
    if not points or len(points) < LANDMARKS:
        return None, None
    ox, oy = points[WRIST][0]*aspect, points[WRIST][1]
    mx, my = points[MIDDLE_MCP][0]*aspect - ox, points[MIDDLE_MCP][1] - oy
    scale = math.hypot(mx, my)
    if scale < 1e-6:
        return None, None
    orientation = math.atan2(my, mx)
    # Rotate the wrist -> middle-MCP vector onto -y, which is "up" in image
    # coordinates where y grows downward.
    angle = -orientation - math.pi/2
    cos, sin = math.cos(angle), math.sin(angle)
    pose = []
    for p in points[:LANDMARKS]:
        x, y = p[0]*aspect - ox, p[1] - oy
        pose.append(((x*cos - y*sin)/scale, (x*sin + y*cos)/scale))
    return tuple(pose), orientation


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

    def matches(self, pose, orientation):
        """Distance if this template accepts the pose, else None."""
        if not self.pose or pose is None:
            return None
        gap = distance(pose, self.pose)
        if gap > self.threshold:
            return None
        if self.orientation is not None and orientation is not None:
            if abs(angle_delta(orientation, self.orientation)) > self.tolerance:
                return None
        return gap


def build(name, samples, orientations=None, **binding):
    """Average recorded samples into a template and size its accept radius.

    The threshold comes from how much the samples varied while the pose was
    held, so a shaky pose gets a roomier radius than a steady one instead of
    every template sharing one guessed constant.
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
    return Template(name=name, pose=mean, threshold=threshold,
                    orientation=orientation, **binding)


@dataclass
class Library:
    templates: list = field(default_factory=list)

    def match(self, pose, orientation):
        """Closest accepting template, or None."""
        best = None
        for template in self.templates:
            gap = template.matches(pose, orientation)
            if gap is not None and (best is None or gap < best[1]):
                best = (template, gap)
        return best[0] if best else None

    def conflict(self, candidate):
        """Existing template a candidate is too close to, if any.

        Checked at record time: a near-duplicate is not a recognition bug that
        shows up later, it is a template that can never win a comparison.
        """
        for template in self.templates:
            if template.name == candidate.name:
                continue
            if not template.pose or not candidate.pose:
                continue
            if distance(candidate.pose, template.pose) < DISTINCT:
                return template
        return None

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
