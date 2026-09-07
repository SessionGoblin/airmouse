import math
import random

import pytest

from airmouse import actions, strokes
from airmouse.config import Settings
from airmouse.controller import Controller
from airmouse.gestures import Features, State

BOUNDS = (0, 0, 1920, 1080)


def line(x0, y0, x1, y1, n=40, jitter=0.0, seed=0):
    rng = random.Random(seed)
    return [(x0+(x1-x0)*i/(n-1) + rng.gauss(0, jitter),
             y0+(y1-y0)*i/(n-1) + rng.gauss(0, jitter)) for i in range(n)]


def circle(n=60, clockwise=True, start=0.0, radius=.2):
    turn = 1 if clockwise else -1
    return [(.5+radius*math.cos(start+turn*2*math.pi*i/(n-1)),
             .5+radius*math.sin(start+turn*2*math.pi*i/(n-1))) for i in range(n)]


def stroke(name, path, **kw):
    return strokes.Stroke(name=name, points=tuple(strokes.canonical(path)), **kw)


def library():
    return strokes.StrokeLibrary([
        stroke('right', line(.3, .5, .7, .5)),
        stroke('left', line(.7, .5, .3, .5)),
        stroke('up', line(.5, .7, .5, .3)),
        stroke('down', line(.5, .3, .5, .7)),
        stroke('circle', circle()),
    ])


def test_a_stroke_is_recognized_redrawn_elsewhere_smaller_and_sloppier():
    got = library().match(strokes.canonical(line(.25, .46, .62, .53, n=33, jitter=.006, seed=7)))
    assert got is not None and got.name == 'right'


def test_speed_does_not_change_the_match():
    """Resampling by arc length is what removes drawing speed: a stroke that
    dawdles at the start would otherwise crowd points there."""
    even = strokes.canonical(line(.3, .5, .7, .5, n=40))
    # Same path, most samples bunched into the first third.
    slow = [(.3 + .13*i/25, .5) for i in range(25)] + [(.43 + .27*i/9, .5) for i in range(10)]
    assert strokes.path_distance(even, strokes.canonical(slow)) < 12


@pytest.mark.parametrize('drawn, expected', [
    (line(.3, .5, .7, .5), 'right'),
    (line(.7, .5, .3, .5), 'left'),
    (line(.5, .7, .5, .3), 'up'),
    (line(.5, .3, .5, .7), 'down'),
])
def test_direction_is_preserved_rather_than_normalized_away(drawn, expected):
    """Classic $1 rotates to the indicative angle, which would collapse all
    four swipes into one straight line."""
    got = library().match(strokes.canonical(drawn))
    assert got is not None and got.name == expected


def test_opposite_directions_score_far_below_the_threshold():
    right = stroke('right', line(.3, .5, .7, .5))
    assert right.compare(strokes.canonical(line(.7, .5, .3, .5))) < .5
    assert right.compare(strokes.canonical(line(.3, .5, .7, .5))) > .95


def test_free_rotation_matches_a_turned_shape_and_says_so():
    upright = stroke('caret', line(.3, .6, .5, .35) + line(.5, .35, .7, .6), free_rotation=True)
    turned = strokes.canonical(line(.4, .3, .65, .5) + line(.65, .5, .4, .7))
    assert upright.compare(turned) > strokes.THRESHOLD
    fixed = stroke('caret', line(.3, .6, .5, .35) + line(.5, .35, .7, .6))
    assert fixed.compare(turned) < upright.compare(turned)


def test_an_unrelated_scribble_matches_nothing():
    scribble = [(.5+.1*math.sin(i*.9), .5+.1*math.cos(i*1.7)) for i in range(40)]
    assert library().match(strokes.canonical(scribble)) is None


def test_a_straight_stroke_is_scaled_uniformly():
    """Scaling a near 1-D stroke into a square stretches its noise to the full
    box height and turns every wobble into signal."""
    wobbly = strokes.canonical(line(.3, .5, .7, .5, n=40, jitter=.002, seed=3))
    clean = strokes.canonical(line(.3, .5, .7, .5, n=40))
    assert strokes.Stroke(name='x', points=tuple(clean)).compare(wobbly) > .9


def test_too_small_or_too_few_points_is_not_a_stroke():
    assert strokes.canonical(line(.5, .5, .505, .505, n=30)) is None    # barely moved
    assert strokes.canonical([(.1, .1), (.9, .9)]) is None              # too few points
    assert strokes.canonical([]) is None


def test_resample_produces_evenly_spaced_points():
    points = strokes.resample(line(.1, .1, .9, .9, n=17), 64)
    assert len(points) == 64
    gaps = [math.dist(a, b) for a, b in zip(points, points[1:])]
    assert max(gaps) - min(gaps) < 1e-6


def test_library_round_trips_and_survives_corruption(tmp_path):
    path = tmp_path/'strokes.json'
    library().save(path)
    loaded = strokes.StrokeLibrary.load(path)
    assert [s.name for s in loaded.strokes] == ['right', 'left', 'up', 'down', 'circle']
    assert loaded.match(strokes.canonical(line(.3, .5, .7, .5))).name == 'right'
    bad = tmp_path/'bad.json'
    bad.write_text('{oops')
    assert strokes.StrokeLibrary.load(bad).strokes == []
    short = tmp_path/'short.json'
    short.write_text('[{"name": "s", "points": [[0, 0]]}]')
    assert strokes.StrokeLibrary.load(short).strokes == []


def test_conflict_flags_two_strokes_that_cannot_be_told_apart():
    lib = library()
    assert lib.conflict(stroke('another right', line(.28, .52, .69, .48))).name == 'right'
    assert lib.conflict(stroke('caret', line(.3, .6, .5, .35) + line(.5, .35, .7, .6))) is None


# --- capture gating ------------------------------------------------------------

def test_capture_collects_only_while_gated():
    c = strokes.StrokeCapture()
    for i in range(20):
        assert c.update(True, (.3+i*.02, .5)) is None
    finished = c.update(False, (.7, .5))
    assert finished is not None and len(finished) == 20


def test_capture_ignores_an_ungated_pointer():
    c = strokes.StrokeCapture()
    for i in range(20):
        assert c.update(False, (.3+i*.02, .5)) is None
    assert c.path == []


def test_capture_discards_a_twitch():
    c = strokes.StrokeCapture()
    c.update(True, (.5, .5))
    c.update(True, (.51, .5))
    assert c.update(False, None) is None        # too few points to be a stroke


# --- controller integration ----------------------------------------------------

def gated(left):
    """Modifier-hand features with the pinch open or closed."""
    return Features((.2, .5), left, .8, False)


def test_drawing_freezes_the_cursor_and_runs_the_bound_action():
    ran = []
    lib = strokes.StrokeLibrary([stroke('right', line(.3, .5, .7, .5),
                                        action='shell', argument='xterm')])
    c = Controller(Settings(), BOUNDS, backend=None, stroke_library=lib,
                   dispatcher=actions.Dispatcher(spawn=ran.append))
    c.enabled = True
    c.aspect = 1.0
    c.machine.armed = True
    now = 0.0
    for i in range(40):                          # trace with the modifier pinched
        now += .033
        state = c.process(Features((.3+i*.01, .5), .8, .8, False), now, gated(.1))
        assert state == State.DRAWING.value
    c.process(Features((.7, .5), .8, .8, False), now+.033, gated(.8))   # release
    assert ran == ['xterm']


def test_an_unrecognized_stroke_runs_nothing_but_is_reported():
    ran = []
    lib = strokes.StrokeLibrary([stroke('right', line(.3, .5, .7, .5),
                                        action='shell', argument='xterm')])
    c = Controller(Settings(), BOUNDS, backend=None, stroke_library=lib,
                   dispatcher=actions.Dispatcher(spawn=ran.append))
    c.enabled = True
    c.aspect = 1.0
    now = 0.0
    for i in range(40):
        now += .033
        c.process(Features((.5+.1*math.sin(i*.9), .5+.1*math.cos(i*1.7)), .8, .8, False),
                  now, gated(.1))
    c.process(Features((.5, .5), .8, .8, False), now+.033, gated(.8))
    assert ran == []
    name, rating = c.last_stroke
    assert rating < strokes.THRESHOLD           # reported for the debug pane


def test_without_a_modifier_hand_pointing_is_untouched():
    """Single-hand use must behave exactly as before."""
    lib = strokes.StrokeLibrary([stroke('right', line(.3, .5, .7, .5),
                                        action='shell', argument='xterm')])
    ran = []
    c = Controller(Settings(), BOUNDS, backend=None, stroke_library=lib,
                   dispatcher=actions.Dispatcher(spawn=ran.append))
    c.enabled = True
    c.machine.armed = True
    now = 0.0
    for i in range(40):
        now += .033
        state = c.process(Features((.3+i*.01, .5), .8, .8, False), now, None)
        assert state != State.DRAWING.value
    assert ran == []


def test_drawing_releases_a_drag_rather_than_smearing_the_window():
    class Backend:
        def __init__(self): self.events = []
        def move(self, x, y): self.events.append('move')
        def down(self): self.events.append('down')
        def up(self): self.events.append('up')
        def right(self): pass
        def scroll(self, steps): pass
        def close(self): pass

    backend = Backend()
    c = Controller(Settings(), BOUNDS, backend=backend,
                   stroke_library=strokes.StrokeLibrary())
    c.enabled = True
    c.machine.armed = True
    c.machine.down = True
    c.process(Features((.5, .5), .8, .8, False), 1.0, gated(.1))
    assert 'up' in backend.events
    assert not c.machine.down


def test_strokes_can_be_switched_off():
    ran = []
    lib = strokes.StrokeLibrary([stroke('right', line(.3, .5, .7, .5),
                                        action='shell', argument='xterm')])
    c = Controller(Settings(strokes=False), BOUNDS, backend=None, stroke_library=lib,
                   dispatcher=actions.Dispatcher(spawn=ran.append))
    c.enabled = True
    c.aspect = 1.0
    c.machine.armed = True
    now = 0.0
    for i in range(40):
        now += .033
        assert c.process(Features((.3+i*.01, .5), .8, .8, False), now,
                         gated(.1)) != State.DRAWING.value
    assert ran == []
