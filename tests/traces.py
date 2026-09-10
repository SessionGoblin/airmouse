"""Deterministic webcam-like control traces, for testing the pointer in closed
loop rather than one frame at a time.

The old tests fed clean, evenly-timed, noise-free frames, which is why they
passed a pointer that had become history-dependent: a scripted hand path with no
jitter never exercises the differentiate-and-reintegrate path that turned
landmark noise into a slow random walk. Everything here therefore carries the
three properties a real 30 FPS webcam stream has and a scripted one does not --
uneven frame timing, per-frame landmark noise, and the occasional single-frame
excursion -- and every trace is seeded, so a failure is reproducible.

Landmark noise is quoted in normalized frame units, which is how it arrives
from the model. On a 1920-wide desktop at the default calibration the mapping is
about 3000 px per unit, so .003 -- a realistic figure for a held hand -- is
about 9 px of movement in the absolute target, against a 2.5 px filter deadzone.
"""
import math
import random

from airmouse.config import Settings
from airmouse.gestures import Features
from airmouse.mapping import AdaptiveEMA, CursorMapper

BOUNDS = (0, 0, 1920, 1080)
S = Settings()
FPS = 30
# Pixels of absolute-target movement per unit of normalized hand movement.
SCALE = (BOUNDS[2]-1)/(1-2*S.margin)*S.sensitivity


class Legacy:
    """The pre-feature pointer path, verbatim, as the compatibility reference.

    Kept as its own class rather than as `precision_assist=False` so a test can
    still tell the two apart if the switch itself ever regresses.
    """
    def __init__(self, bounds=BOUNDS):
        self.bounds = bounds
        self.filter = AdaptiveEMA()

    def target(self, point, settings):
        x, y, width, height = self.bounds
        def normalized(v):
            return min(1, max(0, (((v-settings.margin)/(1-2*settings.margin))-.5)
                              * settings.sensitivity+.5))
        return x + normalized(point[0])*(width-1), y + normalized(point[1])*(height-1)

    def update(self, point, settings, dt, now=None):
        result = self.filter.update(self.target(point, settings), dt,
                                    settings.smoothing, settings.deadzone)
        x, y, w, h = self.bounds
        return min(x+w-1, max(x, result[0])), min(y+h-1, max(y, result[1]))


class Backend:
    """Records what would have been sent to the desktop."""
    def __init__(self):
        self.events = []
        self.moves = []

    def move(self, x, y):
        self.events.append(('move', (x, y)))
        self.moves.append((x, y))

    def down(self): self.events.append(('down',))
    def up(self): self.events.append(('up',))
    def right(self): self.events.append(('right',))
    def scroll(self, steps): self.events.append(('scroll', steps))
    def close(self): pass


def stream(path, duration, seed=0, fps=FPS, timing=.004, noise=.003, spikes=0.0):
    """[(now, point)] for a hand following `path(t)`, as a webcam would deliver it.

    `timing` is the per-frame jitter in the interval, `noise` the per-axis
    landmark noise, and `spikes` the probability of a single-frame excursion of
    the size a landmark swap produces.
    """
    rng = random.Random(seed)
    frames = []
    now = t = 0.0
    while t < duration:
        step = 1/fps + rng.uniform(-timing, timing)
        now += step
        t += step
        x, y = path(t)
        x += rng.gauss(0, noise)
        y += rng.gauss(0, noise)
        if spikes and rng.random() < spikes:
            x += rng.choice((-1, 1))*rng.uniform(.02, .05)
            y += rng.choice((-1, 1))*rng.uniform(.02, .05)
        frames.append((now, (x, y)))
    return frames


# --- hand paths ----------------------------------------------------------------

def still(at=(.5, .5)):
    return lambda t: at


def steady(speed, start=.25, y=.5):
    """Constant speed in normalized units per second, reversing at the edges.

    A triangle rather than a ramp, so a fast trace does not run into the edge of
    the active region and sit there -- which would quietly turn a test of fast
    movement into a test of a stationary hand.
    """
    low, high = .08, .92
    reach = high - low
    def path(t):
        cycle = (start - low + speed*t) % (2*reach)
        return (low + (cycle if cycle <= reach else 2*reach - cycle), y)
    return path


def acquire(y=.5):
    """A fast approach that decelerates onto a target and settles on it -- the
    movement precision assistance exists for."""
    def path(t):
        if t < .8:
            return (.30+.375*t, y)
        if t < 1.6:
            return (.60+.02*(t-.8), y)
        return (.616, y)
    return path


def cycles(period=2.0, slow=.02, fast=.18):
    """Alternating slow work and fast sweeps, returning to the same place, which
    is what exposes an offset that accumulates."""
    def path(t):
        phase, local = divmod(t, period)
        if int(phase) % 2 == 0:
            return (.35+slow*local, .5)
        return (.35+slow*period-fast*local, .5)
    return path


# --- running them --------------------------------------------------------------

def pointer(mapper, frames, settings=S):
    """[(now, point, cursor, absolute)] for a mapper driven through a trace."""
    rows = []
    mapper.filter.reset(mapper.target(frames[0][1], settings))
    previous = frames[0][0] - 1/FPS
    for now, point in frames:
        dt = min(.1, now-previous)
        previous = now
        cursor = mapper.update(point, settings, dt, now)
        rows.append((now, point, cursor, mapper.target(point, settings)))
    return rows


def both(frames, settings=S):
    """The same trace through the current mapper and the legacy one."""
    return (pointer(CursorMapper(BOUNDS), frames, settings),
            pointer(Legacy(), frames, settings))


def wobble(rows, index=2):
    """(peak-to-peak, RMS about the mean, largest single-frame step) in pixels."""
    points = [row[index] for row in rows]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean = (sum(xs)/len(xs), sum(ys)/len(ys))
    rms = math.sqrt(sum(math.dist(p, mean)**2 for p in points)/len(points))
    p2p = max(max(xs)-min(xs), max(ys)-min(ys))
    steps = [math.dist(a, b) for a, b in zip(points, points[1:])]
    return p2p, rms, max(steps)


def deviation(rows):
    """|cursor - absolute target| per frame: the memorylessness measure."""
    return [math.dist(row[2], row[3]) for row in rows]


# --- pinch traces --------------------------------------------------------------

def pinch(seed=0, aim=(.5, .5), settle=1.2, closure=.20, drift=.02, noise=.003,
          angle=-2.2, hold=.5, drag_after=None, drag_speed=.02, abort=False):
    """[(now, point, left, palm)] for aiming steadily and then pinching.

    `drift` is how far the index fingertip -- which is also the pointing
    landmark -- is pulled by the fingers closing, while the knuckle stays put.
    That is the whole of the click-stability problem: the hand has not gone
    anywhere, but the thing the pointer follows has.
    """
    rng = random.Random(seed)
    frames = []
    now = t = 0.0
    total = settle + closure + hold + (drag_after or 0.0)
    dx, dy = math.cos(angle)*drift, math.sin(angle)*drift
    while t < total:
        step = 1/FPS + rng.uniform(-.004, .004)
        now += step
        t += step
        if t < settle:
            progress = 0.0
        elif t < settle+closure:
            progress = (t-settle)/closure
            if abort and progress > .55:
                progress = max(0.0, 1.1-progress*2)
        else:
            progress = 0.0 if abort else 1.0
        left = .85 - (.85-.12)*progress
        hand = aim[0]
        if drag_after is not None and t > settle+closure+hold:
            hand += drag_speed*(t-settle-closure-hold)*10
        point = (hand + dx*progress + rng.gauss(0, noise),
                 aim[1] + dy*progress + rng.gauss(0, noise))
        # The knuckle carries the hand's own movement and none of the pinch
        # geometry, and is a steadier landmark than a fingertip.
        palm = (hand + rng.gauss(0, noise*.7), aim[1] + rng.gauss(0, noise*.7))
        frames.append((now, point, left, palm))
    return frames


def features(point, left, palm=None, right=.85):
    return Features(point, left, right, False, palm=palm)


def press(controller, frames, palm=True):
    """Drive a controller through a pinch trace.

    Returns (rows, commit) where rows is [(now, left, cursor, started)] and
    commit is the index of the frame mouse-down was emitted on.
    """
    rows = []
    commit = None
    for index, (now, point, left, hand) in enumerate(frames):
        was_down = controller.machine.down
        controller.process(features(point, left, hand if palm else None), now)
        if commit is None and not was_down and controller.machine.down:
            commit = index
        rows.append((now, left, controller.target, controller.drag_started))
    return rows, commit


def click_shift(rows, commit, intent=.70):
    """How far the cursor moved because of the pinch, in pixels.

    Returns (onset_to_commit, after_commit_path, after_commit_worst): the first
    is the target the click actually landed on against the one being aimed at,
    the others are how much the cursor wandered once the button was already
    down and nothing but finger geometry was still changing.
    """
    onset = next((i for i, r in enumerate(rows) if r[1] < intent), None)
    if onset is None or commit is None:
        return None
    anchor = rows[max(0, onset-1)][2]
    at_commit = rows[commit][2]
    if anchor is None or at_commit is None:
        return None
    after = [r[2] for r in rows[commit:] if r[2]]
    path = sum(math.dist(a, b) for a, b in zip(after, after[1:]))
    worst = max((math.dist(after[0], p) for p in after), default=0.0)
    return math.dist(anchor, at_commit), path, worst
