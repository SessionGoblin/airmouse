"""A drag stays attached to the hand through pose uncertainty, and only an
opened pinch ever lets go.

The invariant under all of it: tracking loss is not an intentional release.
"""
import math
import random
import statistics

import pytest

from airmouse import gestures
from airmouse.config import Settings
from airmouse.controller import Controller
from airmouse.gestures import Features, GestureMachine, State
from airmouse.mapping import DRAG, AdaptiveEMA, DragEstimator

BOUNDS = (0, 0, 1920, 1080)
S = Settings()
FPS = 30


class Backend:
    def __init__(self):
        self.events = []
        self.moves = []

    def move(self, x, y):
        self.events.append('move')
        self.moves.append((x, y))

    def down(self): self.events.append('down')
    def up(self): self.events.append('up')
    def right(self): self.events.append('right')
    def scroll(self, steps): self.events.append('scroll')
    def close(self): pass


def build(**settings):
    backend = Backend()
    c = Controller(Settings(**settings), BOUNDS, backend=backend)
    c.enabled = True
    c.machine.armed = True
    return c, backend


def hold(pinch=.1):
    """A machine already holding a drag, ready to be told about the next frame."""
    machine = GestureMachine()
    machine.step(Features((.5, .5), .8, .8, False), 0, S, True)
    machine.step(Features((.5, .5), .8, .8, False), .4, S, True)
    machine.step(Features((.5, .5), pinch, .8, False), 1.0, S, True)
    assert machine.step(Features((.5, .5), pinch, .8, False), 1.1, S, True) == [('down',)]
    return machine


def sweep(c, frames=8, speed=.02, start=1.0, x=.22, pinch=.1, settle=3):
    """Pinch, let the pinch settle, then drag rightwards clear of both edges.

    The knuckle travels with the hand, because that is what the drag deadzone is
    now judged on -- a fingertip alone cannot distinguish a hand that has moved
    from a pinch that has closed.
    """
    now = start
    for _ in range(settle):                     # hold still while the pinch forms
        c.process(Features((x, .5), pinch, .8, False, palm=(x, .5)), now)
        now += 1/FPS
    for _ in range(frames):
        x += speed
        c.process(Features((x, .5), pinch, .8, False, palm=(x, .5)), now)
        now += 1/FPS
    return now - 1/FPS, x


# --- entering and staying ------------------------------------------------------

def test_entering_a_drag_still_needs_a_firm_pinch():
    """The hysteresis must only loosen staying, never pressing: a press is what
    a false positive costs a click for."""
    machine = GestureMachine()
    machine.step(Features((.5, .5), .8, .8, False), 0, S, True)
    machine.step(Features((.5, .5), .8, .8, False), .4, S, True)
    loose = Features((.5, .5), S.pinch+.01, .8, False)
    machine.step(loose, 1.0, S, True)
    # Not firm enough to press: it just points, however long it is held.
    assert machine.step(loose, 1.2, S, True) == [('move', loose.point)]
    assert not machine.down
    firm = Features((.5, .5), S.pinch-.01, .8, False)
    machine.step(firm, 2.0, S, True)
    assert machine.step(firm, 2.1, S, True) == [('down',)]


def test_the_exit_threshold_is_wider_than_the_entry_one():
    assert gestures.drag_release(S) > S.release > S.pinch


def test_a_pinch_loosening_past_the_old_threshold_keeps_the_drag():
    """Landmarks go uncertain exactly when a hand is working, and the old
    threshold read that as letting go."""
    machine = hold()
    loosened = Features((.5, .5), S.release+.02, .8, False)
    assert machine.step(loosened, 1.4, S, True) == [('move', loosened.point)]
    assert machine.down


def test_a_weak_pose_is_reported_as_degraded_rather_than_released():
    machine = hold()
    weak = Features((.5, .5), gestures.drag_release(S)-.02, .8, False)
    assert machine.step(weak, 1.4, S, True)[0][0] == 'move'
    assert machine.down
    assert machine.confidence < gestures.DEGRADED_CONFIDENCE
    assert machine.tracking == gestures.TRACKING_DEGRADED


def test_several_degraded_frames_in_a_row_do_not_release():
    machine = hold()
    now = 1.1
    for pinch in (.44, .50, .55, .52, .46, .48):
        now += 1/FPS
        assert machine.step(Features((.5, .5), pinch, .8, False), now, S, True)[0][0] == 'move'
        assert machine.down


def test_clean_tracking_resumes_without_resetting_the_state():
    machine = hold()
    pressed = machine.pressed_at
    machine.step(Features((.5, .5), .55, .8, False), 1.2, S, True)
    machine.step(None, 1.23, S, True)
    machine.step(Features((.5, .5), .1, .8, False), 1.26, S, True)
    assert machine.down
    assert machine.armed
    assert machine.pressed_at == pressed            # the same drag, not a new one
    assert machine.confidence == 1.0
    assert machine.tracking == gestures.TRACKING_OK


def test_opening_the_pinch_releases_immediately():
    """True release stays immediate: no grace period, no confirmation frames."""
    machine = hold()
    assert machine.step(Features((.5, .5), .8, .8, False), 1.15, S, True) == [('up',)]
    assert not machine.down
    assert machine.release_reason == gestures.RELEASE_INTENTIONAL


def test_an_unreadable_hand_latches_then_gives_up_after_the_grace_period():
    machine = hold()
    assert machine.step(None, 1.1+S.drag_grace/2, S, True) == []
    assert machine.down
    assert machine.step(None, 1.1+S.drag_grace+.01, S, True) == [('up',)]
    assert machine.release_reason == gestures.RELEASE_TIMEOUT


def test_the_grace_period_runs_from_the_last_good_frame():
    """Not from the first bad one, or a slow pipeline handing a frame over late
    would buy itself a fresh grace period every time."""
    machine = hold()
    machine.step(None, 5.0, S, True)                # a long stall, noticed once
    assert not machine.down


def test_a_configurable_grace_period():
    machine = hold()
    patient = Settings(drag_grace=.4)
    assert machine.step(None, 1.35, patient, True) == []
    assert machine.down
    assert machine.step(None, 1.55, patient, True) == [('up',)]


def test_pausing_releases_a_latched_drag_on_the_spot():
    """An emergency stop is not a gesture and gets no grace period."""
    machine = hold()
    assert machine.step(None, 1.11, S, False) == [('up',)]
    assert machine.release_reason == gestures.RELEASE_PAUSED


def test_pinch_confidence_spans_the_two_thresholds():
    assert gestures.pinch_confidence(S.pinch, S) == 1.0
    assert gestures.pinch_confidence(.05, S) == 1.0
    assert gestures.pinch_confidence(gestures.drag_release(S), S) == 0.0
    assert gestures.pinch_confidence(.9, S) == 0.0
    assert 0 < gestures.pinch_confidence(S.release, S) < 1
    assert gestures.pinch_confidence(float('nan'), S) == 0.0
    assert gestures.pinch_confidence(None, S) == 0.0


# --- the anchor estimate -------------------------------------------------------

def test_a_steady_drag_tracks_with_less_lag_than_the_shared_filter():
    """Stability must not be bought with latency: that is what makes a dragged
    window feel like it is on a rubber band. Measured against the filter the
    pre-feature drag actually used, on the same input."""
    measurements = [(100.+60*i, 500.) for i in range(30)]   # 1800 px/s at 30 fps
    estimator = DragEstimator()
    estimator.reset(measurements[0])
    for point in measurements:
        estimator.predict(1/FPS)
        estimator.correct(point, 1/FPS, 1.0, 1920)
    shared = AdaptiveEMA()
    shared.reset(measurements[0])
    for point in measurements:
        shared.update(point, 1/FPS, S.smoothing, S.deadzone)
    target = measurements[-1][0]
    assert abs(target-estimator.position[0]) < 10
    assert abs(target-estimator.position[0]) < abs(target-shared.value[0])/3
    assert estimator.velocity[0] == pytest.approx(1800, rel=.15)


def test_a_still_drag_is_steadier_than_the_shared_filter():
    """The other half of the same trade. A fixed alpha cannot win both, which
    is why alpha is chosen from the size of the residual."""
    rng = random.Random(29)
    noisy = [(800+rng.gauss(0, 9), 500+rng.gauss(0, 9)) for _ in range(120)]
    estimator = DragEstimator()
    estimator.reset(noisy[0])
    tracked = []
    for point in noisy:
        estimator.predict(1/FPS)
        tracked.append(estimator.correct(point, 1/FPS, 1.0, 1920)[0])
    shared = AdaptiveEMA()
    shared.reset(noisy[0])
    smoothed = [shared.update(p, 1/FPS, S.smoothing, S.deadzone) for p in noisy]
    def wobble(path):
        return statistics.fmean([math.dist(a, b) for a, b in zip(path, path[1:])])
    assert wobble(tracked) < wobble(smoothed)


def test_a_fast_drag_cannot_lock_itself_out_of_tracking():
    """A rejected frame used to freeze the tracked velocity, a frozen velocity
    froze the outlier limit, and the drag never recovered -- it could not even
    throw, because every sample stayed unclean."""
    estimator = DragEstimator()
    estimator.reset((0., 500.))
    step = 15000/FPS                                # far faster than any hand
    for i in range(30):
        estimator.predict(1/FPS)
        estimator.correct((step*(i+1), 500.), 1/FPS, 1.0, 1920)
    assert abs(step*30 - estimator.position[0]) < 100
    assert not estimator.rejected


def test_an_implausible_excursion_is_clamped_not_followed():
    estimator = DragEstimator()
    estimator.reset((500., 500.))
    for i in range(6):
        estimator.predict(1/FPS)
        estimator.correct((500.+40*(i+1), 500.), 1/FPS, 1.0, 1920)
    before = estimator.position[0]
    estimator.predict(1/FPS)
    _, accepted = estimator.correct((before+900, 500.), 1/FPS, 1.0, 1920)
    assert not accepted
    assert estimator.position[0] - before < 150     # a fraction of the excursion


def test_low_confidence_leans_on_the_trajectory():
    """Two identical measurements, two confidences: the uncertain one must move
    the estimate less, which is the whole of confidence-aware blending."""
    def run(confidence):
        estimator = DragEstimator()
        estimator.reset((500., 500.))
        estimator.predict(1/FPS)
        estimator.correct((530., 500.), 1/FPS, confidence, 1920)
        return estimator.position[0]
    assert run(1.0) - 500 > (run(.05) - 500) * 2


def test_a_reacquired_hand_is_converged_onto_rather_than_snapped_to():
    estimator = DragEstimator()
    estimator.reset((500., 500.))
    estimator.predict(1/FPS, blind=True)
    first = estimator.correct((700., 500.), 1/FPS, 1.0, 1920)[0][0]
    assert first < 700                              # not there yet
    assert estimator.velocity[0] == pytest.approx(0, abs=1)   # and no fake speed
    seen = []
    for _ in range(20):
        estimator.predict(1/FPS)
        seen.append(estimator.correct((700., 500.), 1/FPS, 1.0, 1920)[0][0])
    assert max(seen) < 700 + 40                     # bounded overshoot
    # The last few pixels close at the noise-rejection rate rather than the
    # tracking rate, deliberately: by then the residual is the size of landmark
    # noise, and chasing it is what made a held drag shimmer.
    assert estimator.position[0] == pytest.approx(700, abs=15)


def test_a_reacquisition_contributes_no_velocity():
    """The safety invariant behind it: a hand that was merely re-found is not a
    hand travelling at the speed of the gap, and a throw must never be able to
    be assembled out of one."""
    estimator = DragEstimator()
    estimator.reset((500., 500.))
    for _ in range(4):
        estimator.predict(1/FPS, blind=True)
    estimator.correct((900., 500.), 1/FPS, 1.0, 1920)
    assert math.hypot(*estimator.velocity) < 60     # a 400 px jump, no speed


def test_blind_extrapolation_is_damped():
    """A latched drag drifting on a stale velocity would fly off if the hand
    really has gone."""
    estimator = DragEstimator()
    estimator.reset((500., 500.))
    for i in range(8):
        estimator.predict(1/FPS)
        estimator.correct((500.+60*(i+1), 500.), 1/FPS, 1.0, 1920)
    start = estimator.position[0]
    for _ in range(5):
        estimator.predict(1/FPS, blind=True)
    undamped = estimator.velocity[0]
    assert 0 < estimator.position[0]-start < 1800*5/FPS
    assert undamped < 1800


# --- through the controller ----------------------------------------------------

def test_a_dropped_frame_keeps_the_window_moving():
    c, backend = build()
    now, x = sweep(c)
    before = c.target[0]
    now += 1/FPS
    state = c.process(None, now)
    assert state == State.DRAGGING.value
    assert c.machine.down
    assert backend.events.count('up') == 0
    assert c.target[0] > before                     # coasted, rather than froze
    assert c.telemetry.latched


def test_a_latched_drag_survives_and_then_resumes():
    c, backend = build()
    now, x = sweep(c)
    for _ in range(3):                              # three unreadable frames
        now += 1/FPS
        c.process(None, now)
    for _ in range(4):
        now += 1/FPS
        x += .02
        c.process(Features((x, .5), .1, .8, False), now)
    assert c.machine.down
    assert c.drag_started
    assert backend.events.count('up') == 0
    assert not c.telemetry.latched
    assert c.telemetry.drag_quality > .5


def test_a_loosening_pinch_does_not_drop_the_drag_mid_sweep():
    c, backend = build()
    now, x = sweep(c)
    for pinch in (.45, .52, .55, .48):
        now += 1/FPS
        x += .02
        c.process(Features((x, .5), pinch, .8, False), now)
    assert c.machine.down
    assert backend.events.count('up') == 0


def test_a_prolonged_loss_ends_the_drag_without_throwing():
    c, backend = build()
    now, x = sweep(c)
    now += S.drag_grace + .05
    c.process(None, now)
    assert not c.machine.down
    assert c.throw is None
    assert backend.events.count('up') == 1
    assert c.machine.release_reason == gestures.RELEASE_TIMEOUT


def test_an_isolated_outlier_cannot_teleport_the_window():
    c, backend = build()
    now, x = sweep(c, speed=.01, frames=12)         # a deliberate, established drag
    assert c.drag_started
    steady = c.target[0]
    now += 1/FPS
    # ~600 px of excursion in the pointing landmark, with the hand where it was.
    c.process(Features((x+.2, .5), .1, .8, False, palm=(x, .5)), now)
    assert c.target[0] - steady < 150
    assert c.telemetry.outlier


# --- safety --------------------------------------------------------------------

def test_tracking_loss_never_throws_however_fast_the_drag_was():
    c, backend = build()
    now, x = sweep(c, frames=10, speed=.05)         # fast enough to earn a throw
    assert c.fling_velocity(now) is not None
    now += S.drag_grace + .05
    c.process(None, now)
    assert c.throw is None
    assert backend.events.count('up') == 1


def test_a_reacquisition_cannot_manufacture_a_fling():
    """The frames either side of a gap are smooth motion the user never made."""
    c, backend = build()
    now, x = sweep(c, speed=.002)                   # too slow to earn a throw
    now += 1/FPS
    c.process(None, now)
    now += 1/FPS
    x += .15                                        # reacquired 450 px away
    c.process(Features((x, .5), .1, .8, False), now)
    now += 1/FPS
    c.process(Features((x, .5), .8, .8, False), now)
    assert c.throw is None
    velocity = c.telemetry.release_velocity
    assert velocity is None or math.hypot(*velocity) < 600


def test_predicted_positions_take_no_part_in_a_release_velocity():
    c, _ = build()
    now, x = sweep(c, speed=.01, frames=12)
    assert c.drag_started
    now += 1/FPS
    c.process(None, now)                            # emits a predicted position
    assert any(not clean for _, _, clean in c.trail)
    clean = [(t, p) for t, p, ok in c.trail if ok]
    assert len(clean) < len(c.trail)


def test_an_outlier_cannot_set_the_direction_of_a_throw():
    """One glitched sample in the window must move the answer by nothing, which
    is why the estimate is a median rather than the window's endpoints."""
    c, _ = build()
    now = 10.0
    for i in range(9):
        c.record_trail(now - .018*(9-i), (100 + i*40, 500))
    steady = c.fling_velocity(now)
    c.record_trail(now - .07, (100 + 4*40, 3000))   # one wild sample, mid-window
    spiked = c.fling_velocity(now)
    assert abs(spiked[1]) < abs(steady[1]) + 400
    assert spiked[0] == pytest.approx(steady[0], rel=.35)


def test_a_release_velocity_needs_more_than_the_final_frame_pair():
    c, _ = build()
    c.record_trail(9.9, (100, 500))
    c.record_trail(9.93, (700, 500))
    assert c.fling_velocity(9.94) is None           # inside SETTLE, and too short


def test_the_throw_records_why_it_was_accepted_or_refused():
    c, _ = build()
    now, x = sweep(c, speed=.004, frames=14)        # moves, but far too slowly
    c.process(Features((x, .5), .8, .8, False, palm=(x, .5)), now + 1/FPS)
    assert c.telemetry.release_reason == gestures.RELEASE_INTENTIONAL
    assert 'rejected' in c.telemetry.throw
    c, _ = build()
    now, x = sweep(c, frames=12)
    c.process(Features((x, .5), .8, .8, False, palm=(x, .5)), now + 1/FPS)
    assert c.telemetry.throw == 'accepted'


def test_telemetry_stays_readable_with_no_hand_at_all():
    c, _ = build()
    assert c.telemetry.lines()
    c.process(None, 1.0)
    assert isinstance('\n'.join(c.telemetry.lines()), str)
