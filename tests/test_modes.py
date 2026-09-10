"""Modifier poses as modes, and the chord scoping they enable."""
import random

import pytest

from airmouse import actions, poses, strokes
from airmouse.config import Settings
from airmouse.controller import Controller
from airmouse.gestures import Features, State

BOUNDS = (0, 0, 1920, 1080)
S = Settings()


def shape(seed):
    rng = random.Random(seed)
    points = [(rng.uniform(.3, .7), rng.uniform(.2, .8), 0.) for _ in range(21)]
    points[poses.WRIST] = (.5, .8, 0.)
    points[poses.MIDDLE_MCP] = (.5, .5, 0.)
    return points


def features(seed, left=.8):
    pose, orientation, chirality = poses.normalize(shape(seed), 1.0)
    return Features((.5, .5), left, .8, False, pose, orientation, chirality)


def template(name, seed, **kw):
    pose, _, _ = poses.normalize(shape(seed), 1.0)
    return poses.Template(name=name, pose=pose, threshold=.3, **kw)


def controller(templates=(), stroke_list=(), spawn=None):
    c = Controller(S, BOUNDS, backend=None,
                   library=poses.Library(list(templates)),
                   stroke_library=strokes.StrokeLibrary(list(stroke_list)),
                   dispatcher=actions.Dispatcher(spawn=spawn or (lambda x: None)))
    c.enabled = True
    c.aspect = 1.0
    c.machine.armed = True
    return c


def hold(c, pointer, modifier, frames=25, start=1.0):
    """Run frames at 30 fps so dwell elapses without tripping the lost-hand reset."""
    state = None
    for i in range(frames):
        state = c.process(pointer, start + i/30, modifier)
    return state


# --- modes ---------------------------------------------------------------------

def test_a_held_modifier_pose_becomes_the_mode():
    c = controller([template('fist', 1, role='modifier')])
    hold(c, features(5), features(1))
    assert c.mode == 'fist'


def test_releasing_the_modifier_clears_the_mode():
    c = controller([template('fist', 1, role='modifier')])
    hold(c, features(5), features(1))
    hold(c, features(5), features(9), start=2.0)
    assert c.mode is None


def test_a_modifier_pose_fires_its_binding_once():
    ran = []
    c = controller([template('fist', 1, role='modifier', action='shell', argument='xterm')],
                   spawn=ran.append)
    hold(c, features(5), features(1), frames=60)
    assert ran == ['xterm']          # latched, not once per frame


def test_a_pointer_pose_is_not_read_from_the_modifier_hand():
    """Roles scope matching, so the same shape can sit on both hands."""
    ran = []
    c = controller([template('wave', 1, role='pointer', action='shell', argument='pointer'),
                    template('wave-mod', 1, role='modifier', action='shell', argument='modifier')],
                   spawn=ran.append)
    hold(c, features(7), features(1))     # only the modifier makes the shape
    assert ran == ['modifier']


# --- chord scoping -------------------------------------------------------------

def test_a_scoped_pointer_pose_only_fires_while_its_mode_is_held():
    ran = []
    c = controller([template('fist', 1, role='modifier'),
                    template('chord', 2, when='fist', action='shell', argument='chord')],
                   spawn=ran.append)
    hold(c, features(2), features(9), frames=45)   # right pointer pose, wrong mode
    assert ran == []
    hold(c, features(2), features(1), frames=45, start=3.0)
    assert ran == ['chord']


def test_a_chord_costs_two_dwells_in_sequence():
    """The modifier has to confirm before the scoped pose is even eligible, so
    a chord takes about twice custom_dwell. Slower than a plain pose, and the
    price of not firing chords on a modifier shape passing through."""
    ran = []
    c = controller([template('fist', 1, role='modifier'),
                    template('chord', 2, when='fist', action='shell', argument='chord')],
                   spawn=ran.append)
    one = int(S.custom_dwell * 30) + 2
    hold(c, features(2), features(1), frames=one)          # one dwell only
    assert ran == []
    hold(c, features(2), features(1), frames=one*2 + 2)    # two dwells
    assert ran == ['chord']


def test_an_unscoped_pose_still_fires_while_a_mode_is_held():
    """Holding a modifier must not switch off every ordinary gesture."""
    ran = []
    c = controller([template('fist', 1, role='modifier'),
                    template('plain', 2, action='shell', argument='plain')],
                   spawn=ran.append)
    hold(c, features(2), features(1))
    assert ran == ['plain']


def test_a_scoped_pose_outranks_an_unscoped_one_of_the_same_shape():
    """The more specific binding is the one the modifier was raised for."""
    library = poses.Library([template('fist', 1, role='modifier'),
                             template('plain', 2, action='shell', argument='plain'),
                             template('chord', 2, when='fist', action='shell', argument='chord')])
    pointer = features(2)
    assert library.match(pointer.pose, pointer.orientation, pointer.chirality,
                         mode='fist').name == 'chord'
    assert library.match(pointer.pose, pointer.orientation, pointer.chirality,
                         mode=None).name == 'plain'


def test_same_shape_on_different_hands_is_not_a_conflict():
    """Recording a modifier fist must not warn that it collides with a pointer
    fist: they are read from different hands and can never compete."""
    library = poses.Library([template('fist', 1, role='pointer')])
    assert library.conflict(template('fist-mod', 1, role='modifier')) is None
    assert library.conflict(template('fist-two', 1, role='pointer')).name == 'fist'


def test_same_shape_under_different_modes_is_not_a_conflict():
    library = poses.Library([template('a', 1, when='one')])
    assert library.conflict(template('b', 1, when='two')) is None
    assert library.conflict(template('c', 1, when='one')).name == 'a'


# --- drawing gate --------------------------------------------------------------

def line(x0, y0, x1, y1, n=40):
    return [(x0+(x1-x0)*i/(n-1), y0+(y1-y0)*i/(n-1)) for i in range(n)]


def drawn(name, path, **kw):
    return strokes.Stroke(name=name, points=tuple(strokes.canonical(path)), **kw)


def test_the_pinch_gates_drawing_until_a_gate_pose_exists():
    ran = []
    c = controller([], [drawn('right', line(.3, .5, .7, .5), action='shell', argument='swipe')],
                   spawn=ran.append)
    now = 0.0
    for i in range(40):
        now += .033
        c.process(Features((.3+i*.01, .5), .8, .8, False), now, Features((.2, .5), .1, .8, False))
    c.process(Features((.7, .5), .8, .8, False), now+.033, Features((.2, .5), .8, .8, False))
    assert ran == ['swipe']


def test_a_gate_pose_replaces_the_pinch():
    ran = []
    c = controller([template('fist', 1, role='modifier', gates_drawing=True)],
                   [drawn('right', line(.3, .5, .7, .5), action='shell', argument='swipe')],
                   spawn=ran.append)
    # A bare pinch no longer draws once a gate pose is recorded.
    now = 0.0
    for i in range(40):
        now += .033
        state = c.process(Features((.3+i*.01, .5), .8, .8, False), now,
                          Features((.2, .5), .1, .8, False))
        assert state != State.DRAWING.value
    assert ran == []


def test_a_gate_pose_does_not_also_run_its_own_binding():
    """A gate is a mode, not an event; firing its action would run it every
    time drawing starts."""
    ran = []
    c = controller([template('fist', 1, role='modifier', gates_drawing=True,
                             action='shell', argument='oops')], spawn=ran.append)
    hold(c, features(5), features(1), frames=60)
    assert ran == []


def test_strokes_are_scoped_to_the_gate_that_was_held():
    ran = []
    c = controller([template('fist', 1, role='modifier', gates_drawing=True),
                    template('peace', 3, role='modifier', gates_drawing=True)],
                   [drawn('right', line(.3, .5, .7, .5), when='fist',
                          action='shell', argument='under-fist'),
                    drawn('right-peace', line(.3, .5, .7, .5), when='peace',
                          action='shell', argument='under-peace')],
                   spawn=ran.append)
    now = 0.0
    gate = features(3)                       # hold "peace" as the gate
    for i in range(40):
        now += .033
        c.process(Features((.3+i*.01, .5), .8, .8, False), now, gate)
    c.process(Features((.7, .5), .8, .8, False), now+.033, features(9))
    assert ran == ['under-peace']            # same shape, other gate's binding


def test_the_mode_survives_until_the_stroke_is_recognized():
    """Recognition runs on the frame the gate opens, when the mode has already
    cleared -- the mode held during drawing is what must be used."""
    ran = []
    c = controller([template('fist', 1, role='modifier', gates_drawing=True)],
                   [drawn('right', line(.3, .5, .7, .5), when='fist',
                          action='shell', argument='scoped')],
                   spawn=ran.append)
    now = 0.0
    for i in range(40):
        now += .033
        c.process(Features((.3+i*.01, .5), .8, .8, False), now, features(1))
    c.process(Features((.7, .5), .8, .8, False), now+.033, features(9))
    assert c.mode is None                    # gate released by recognition time
    assert ran == ['scoped']


# --- latch ---------------------------------------------------------------------

def test_pose_latch_fires_once_and_re_arms_only_after_release():
    latch = poses.PoseLatch()
    a = template('a', 1)
    assert latch.update(a, 0.0, .2) is None          # confirming
    assert latch.update(a, .1, .2) is None
    assert latch.update(a, .25, .2) is a             # fires
    assert latch.update(a, .5, .2) is None           # held, not repeated
    assert latch.template is a                       # still the mode
    assert latch.update(None, .6, .2) is None        # released
    assert latch.template is None
    assert latch.update(a, .7, .2) is None
    assert latch.update(a, .95, .2) is a             # fires again


def test_pose_latch_restarts_the_dwell_on_a_different_pose():
    latch = poses.PoseLatch()
    a, b = template('a', 1), template('b', 2)
    latch.update(a, 0.0, .2)
    assert latch.update(b, .15, .2) is None          # swapped; dwell restarts
    assert latch.update(b, .3, .2) is None           # not yet .2 since the swap
    assert latch.update(b, .4, .2) is b


# --- dialog isolation ----------------------------------------------------------

def test_editing_a_gesture_does_not_reach_the_live_library_until_saved(tmp_path, monkeypatch):
    """The dialog shared Template objects with the library the controller was
    matching against, so an edit took effect immediately and cancelling could
    not undo it. Flipping a pose to a drawing gate that way froze the cursor."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from airmouse import recorder
    QApplication.instance() or QApplication([])
    monkeypatch.setattr('airmouse.poses.GESTURE_PATH', tmp_path/'gestures.json')

    live = poses.Library([template('wave', 2, action='shell', argument='xterm')])
    d = recorder.GestureDialog(live, S, lambda: ([], None))
    assert d.library.templates[0] is not live.templates[0]

    d.list.setCurrentRow(0)
    d.role.setCurrentIndex(d.role.findData('modifier'))
    d.gates.setChecked(True)
    d.kind.setCurrentIndex(d.kind.findData('none'))
    assert live.templates[0].role == 'pointer'          # untouched
    assert live.templates[0].action == 'shell'
    assert live.gates() == set()                        # cursor cannot be frozen by this
    d.reject()
    assert live.templates[0].action == 'shell'


def test_the_hand_can_be_reassigned_without_re_recording(tmp_path, monkeypatch):
    """Poses are stored mirrored into one chirality, so the recording is valid
    for either hand; only which hand it is read from changes."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from airmouse import recorder
    QApplication.instance() or QApplication([])
    monkeypatch.setattr('airmouse.poses.GESTURE_PATH', tmp_path/'gestures.json')

    d = recorder.GestureDialog(poses.Library([template('wave', 2)]), S, lambda: ([], None))
    d.list.setCurrentRow(0)
    before = d.current().pose
    d.role.setCurrentIndex(d.role.findData('modifier'))
    assert d.current().role == 'modifier'
    assert d.current().pose == before                   # no re-recording needed
    d.apply()
    assert poses.Library.load(tmp_path/'gestures.json').templates[0].role == 'modifier'


def test_a_deleted_gate_stops_gating_once_applied():
    """The reported symptom: the modifier kept blocking after its gate pose was
    removed. A gate suppresses the pinch fallback, so a stale one freezes the
    cursor whenever the second hand is up."""
    gate = template('gate', 1, role='modifier', gates_drawing=True)
    c = controller([gate])
    assert c.machine.library.gates() == {'gate'}
    hold(c, features(5), features(1))
    assert c.drawing_gate(features(1)) is True
    # What the window does when the dialog saves.
    c.machine.library = poses.Library([])
    c.mode = None
    assert c.drawing_gate(features(1)) is False          # falls back to the pinch
    assert c.drawing_gate(features(1, left=.1)) is True
