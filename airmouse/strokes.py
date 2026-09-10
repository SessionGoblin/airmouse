"""$1 unistroke recognition for paths drawn with the pointer hand.

Wobbrock, Wilson and Li (2007): resample to a fixed number of points, scale,
translate to the origin, then compare point-for-point under a small rotation
search. Pure geometry and a JSON store, so recognition is testable without a
camera.

Segmentation -- knowing where a stroke starts and ends -- is the hard half of
path recognition, and it is not solved here: the modifier hand gates it. That is
deliberate. This app's pointer *is* the hand, so a recognizer running freely
over the cursor path fires constantly during ordinary pointing.
"""
from dataclasses import dataclass, asdict, field
import json
import math
from pathlib import Path

STROKE_PATH = Path.home() / '.config' / 'airmouse' / 'strokes.json'

RESAMPLE = 64          # points every stroke is resampled to
SQUARE = 250.0         # reference box strokes are scaled into
MIN_POINTS = 8         # fewer than this is a twitch, not a stroke
MIN_TRAVEL = .06       # normalized frame widths; rejects a stationary hand
THRESHOLD = .78        # $1 score below which nothing matched
# A stroke thinner than this fraction of its length is effectively 1-D. Scaling
# such a stroke into a square multiplies its noise up to the full box height and
# turns every wobble in a straight swipe into signal, so those scale uniformly.
THIN = .25
# Rotation search. Direction-sensitive strokes still allow some slop, because a
# swipe is never drawn at exactly the angle it was recorded at.
SEARCH_FREE = math.radians(45)
SEARCH_FIXED = math.radians(15)
ANGLE_PRECISION = math.radians(2)
PHI = .5 * (-1 + math.sqrt(5))


def path_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def resample(points, n=RESAMPLE):
    """Re-space a path to n points at equal arc length.

    Equal spacing is what makes the comparison independent of how fast the
    stroke was drawn: a slow start would otherwise crowd points into the
    beginning and dominate the distance.
    """
    total = path_length(points)
    if total <= 0:
        return [tuple(points[0])] * n
    interval = total / (n - 1)
    output = [tuple(points[0])]
    remaining = list(points)
    carried = 0.0
    index = 1
    while index < len(remaining):
        previous, current = remaining[index-1], remaining[index]
        step = math.dist(previous, current)
        if step <= 0:
            index += 1
            continue
        if carried + step >= interval:
            ratio = (interval - carried) / step
            split = (previous[0] + ratio*(current[0]-previous[0]),
                     previous[1] + ratio*(current[1]-previous[1]))
            output.append(split)
            remaining.insert(index, split)
            carried = 0.0
        else:
            carried += step
        index += 1
    while len(output) < n:
        output.append(tuple(remaining[-1]))
    return output[:n]


def centroid(points):
    n = len(points)
    return (sum(p[0] for p in points)/n, sum(p[1] for p in points)/n)


def rotate(points, angle, about=None):
    about = about or centroid(points)
    cos, sin = math.cos(angle), math.sin(angle)
    return [((p[0]-about[0])*cos - (p[1]-about[1])*sin + about[0],
             (p[0]-about[0])*sin + (p[1]-about[1])*cos + about[1]) for p in points]


def indicative_angle(points):
    """Angle from the centroid to the first point."""
    middle = centroid(points)
    return math.atan2(middle[1]-points[0][1], middle[0]-points[0][0])


def scale(points, size=SQUARE):
    """Fit into a reference box, uniformly when the stroke is effectively 1-D."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width, height = max(xs)-min(xs), max(ys)-min(ys)
    longest = max(width, height)
    if longest <= 0:
        return [(0.0, 0.0) for _ in points]
    if min(width, height) < longest * THIN:
        factor = size / longest
        return [(p[0]*factor, p[1]*factor) for p in points]
    return [(p[0]*size/width, p[1]*size/height) for p in points]


def translate_to_origin(points):
    middle = centroid(points)
    return [(p[0]-middle[0], p[1]-middle[1]) for p in points]


def canonical(path, aspect=4/3):
    """Raw fingertip path -> comparable point list, or None if too slight.

    Direction is preserved. Classic $1 rotates every stroke to its indicative
    angle here, which is what makes it rotation invariant -- and what would
    turn swipe-left, swipe-right and swipe-up into one straight line. Strokes
    that genuinely want orientation invariance opt in, and get it at compare
    time instead.
    """
    if not path or len(path) < MIN_POINTS:
        return None
    stretched = [(p[0]*aspect, p[1]) for p in path]
    if path_length(stretched) < MIN_TRAVEL:
        return None
    return translate_to_origin(scale(resample(stretched)))


def to_indicative(points):
    """Rotate a centred stroke so its heading points a fixed way.

    Applied to both sides of a comparison, this is what makes an orientation
    free stroke match at any angle: a bare widening of the search bound only
    reaches the ends of that bound, so a shape turned by 90 degrees stays out
    of range however generous it is.
    """
    return rotate(points, -indicative_angle(points), about=(0.0, 0.0))


def rejection(path, aspect=4/3):
    """Why canonical() would refuse this path, or None if it would accept it.

    A bare "too short" cannot be acted on: too few points means the gate
    dropped mid-stroke, too little travel means the shape was drawn small.
    """
    if not path or len(path) < MIN_POINTS:
        return (f'only {len(path or ())} points captured — hold the gate steady through the '
                'whole stroke')
    if path_length([(p[0]*aspect, p[1]) for p in path]) < MIN_TRAVEL:
        return 'the hand barely moved — draw the shape larger'
    return None


def path_distance(a, b):
    return sum(math.dist(p, q) for p, q in zip(a, b)) / len(a)


def distance_at_angle(points, template, angle):
    return path_distance(rotate(points, angle, about=(0.0, 0.0)), template)


def best_distance(points, template, bound):
    """Golden-section search for the rotation that fits best."""
    if bound <= 0:
        return distance_at_angle(points, template, 0.0)
    low, high = -bound, bound
    x1 = PHI*low + (1-PHI)*high
    f1 = distance_at_angle(points, template, x1)
    x2 = (1-PHI)*low + PHI*high
    f2 = distance_at_angle(points, template, x2)
    while abs(high-low) > ANGLE_PRECISION:
        if f1 < f2:
            high, x2, f2 = x2, x1, f1
            x1 = PHI*low + (1-PHI)*high
            f1 = distance_at_angle(points, template, x1)
        else:
            low, x1, f1 = x1, x2, f2
            x2 = (1-PHI)*low + PHI*high
            f2 = distance_at_angle(points, template, x2)
    return min(f1, f2)


def score(points, template, bound):
    """$1 similarity in 0..1; the half-diagonal is the worst possible fit."""
    return 1 - best_distance(points, template, bound) / (.5 * math.hypot(SQUARE, SQUARE))


@dataclass
class Stroke:
    name: str
    action: str = ''
    argument: str = ''
    points: tuple = ()
    # True matches the shape at any orientation, which also merges strokes that
    # differ only in direction. False keeps left, right and up distinct.
    free_rotation: bool = False
    # Only match while this modifier pose gated the stroke. Empty matches
    # whichever gate was held, so a single-gate setup needs no scoping.
    when: str = ''

    def compare(self, points):
        if not self.points or points is None:
            return 0.0
        template = list(self.points)
        if self.free_rotation:
            points, template = to_indicative(points), to_indicative(template)
            return score(points, template, SEARCH_FREE)
        return score(points, template, SEARCH_FIXED)


@dataclass
class StrokeLibrary:
    strokes: list = field(default_factory=list)

    def match(self, points, threshold=THRESHOLD, mode=None):
        """Best stroke drawn under the given gate, or None.

        A stroke scoped to the held gate outranks an unscoped one, so the same
        shape can mean different things under different modifier poses.
        """
        best, rank = None, None
        for stroke in self.strokes:
            if stroke.when and stroke.when != mode:
                continue
            value = stroke.compare(points)
            if value < threshold:
                continue
            candidate = (0 if stroke.when else 1, -value)
            if rank is None or candidate < rank:
                best, rank = stroke, candidate
        return best

    def nearest(self, points):
        """Closest stroke and its score, ignoring the threshold. Diagnostics."""
        if points is None or not self.strokes:
            return None, 0.0
        scored = [(s, s.compare(points)) for s in self.strokes]
        return max(scored, key=lambda pair: pair[1])

    def conflict(self, candidate):
        """An existing stroke this one is too close to to be told apart."""
        for stroke in self.strokes:
            if stroke.name == candidate.name or not stroke.points:
                continue
            if stroke.when != candidate.when:
                continue        # different gates; the same shape can mean two things
            if stroke.compare(list(candidate.points)) >= THRESHOLD:
                return stroke
        return None

    def replace(self, stroke):
        self.strokes = [s for s in self.strokes if s.name != stroke.name]
        self.strokes.append(stroke)

    def remove(self, name):
        self.strokes = [s for s in self.strokes if s.name != name]

    @classmethod
    def load(cls, path=None):
        path = Path(path or STROKE_PATH)
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls()
        found = []
        for entry in raw if isinstance(raw, list) else []:
            try:
                stroke = Stroke(**entry)
                stroke.points = tuple(tuple(float(v) for v in p) for p in stroke.points)
                if len(stroke.points) == RESAMPLE:
                    found.append(stroke)
            except (TypeError, ValueError):
                continue
        return cls(found)

    def save(self, path=None):
        path = Path(path or STROKE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps([asdict(s) for s in self.strokes], indent=2))
        temporary.replace(path)


class StrokeCapture:
    """Collects the pointer path while the modifier hand gates it."""

    def __init__(self):
        self.path = []
        self.drawing = False

    def reset(self):
        self.path = []
        self.drawing = False

    def update(self, gating, point, now=None):
        """Feed one frame. Returns the finished raw path when the gate opens."""
        if gating:
            self.drawing = True
            if point is not None:
                self.path.append(tuple(point[:2]))
            return None
        if not self.drawing:
            return None
        finished, self.path = self.path, []
        self.drawing = False
        return finished if len(finished) >= MIN_POINTS else None
