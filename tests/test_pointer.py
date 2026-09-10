"""The pointer, measured on webcam-like traces against the pre-feature path.

The pre-feature pointer felt good and is the thing to protect. So the tests here
are mostly comparative: the same noisy, unevenly-timed trace goes through both
mappers and the new one has to be no worse. That is a stronger contract than any
absolute threshold, and it is the contract the previous test suite lacked --
which is how it certified a pointer whose position depended on its history.
"""
import math
import statistics

import pytest

from airmouse.config import Settings
from airmouse.mapping import PRECISION, CursorMapper, PrecisionClutch, VelocityEstimator
from traces import (BOUNDS, FPS, S, SCALE, Legacy, acquire, both, cycles, deviation,
                    pointer, steady, still, stream, wobble)

# Speeds spanning deliberate work to a fast sweep, in normalized units/s.
NORMAL = (.25, .45, .8, 1.4, 2.2)


# --- legacy compatibility ------------------------------------------------------

@pytest.mark.parametrize('speed', NORMAL)
def test_normal_and_fast_movement_matches_the_legacy_pointer(speed):
    """Not one nominal velocity: across the whole range of ordinary movement the
    pointer has to be the pre-feature pointer, because precision assistance is
    not engaged at any of these speeds."""
    frames = stream(steady(speed), 4.0, seed=int(speed*100))
    new, legacy = both(frames)
    gap = [math.dist(n[2], l[2]) for n, l in zip(new, legacy)]
    assert max(gap) < 1e-9


@pytest.mark.parametrize('speed', NORMAL)
def test_precision_never_engages_at_ordinary_speeds(speed):
    mapper = CursorMapper(BOUNDS)
    rows = pointer(mapper, stream(steady(speed), 4.0, seed=7))
    assert not mapper.precision.active
    assert mapper.precision.offset == (0.0, 0.0)


def test_the_switch_restores_the_legacy_path_exactly():
    plain = Settings(precision_assist=False)
    frames = stream(acquire(), 3.0, seed=3)
    new = pointer(CursorMapper(BOUNDS), frames, plain)
    legacy = pointer(Legacy(), frames, plain)
    assert [r[2] for r in new] == [r[2] for r in legacy]


# --- no persistent offset ------------------------------------------------------

def test_a_fast_sweep_then_a_stop_leaves_no_offset():
    """The failure this replaced: a fast sweep parked up to 154 px of lead on
    the cursor, which then sat there. Stopping must leave the cursor on the
    plain absolute map."""
    def path(t):
        return (.30+.55*t, .5) if t < .6 else (.63, .5)
    frames = stream(path, 4.0, seed=5)
    mapper = CursorMapper(BOUNDS)
    rows = pointer(mapper, frames)
    legacy = pointer(Legacy(), frames)
    settled = [i for i, row in enumerate(rows) if row[0] > 1.2]
    # Against the legacy pointer on the same frames, because the deviation of
    # either from the *instantaneous* absolute target is mostly the filter lag
    # and the landmark noise, which both share.
    mine = statistics.fmean([deviation(rows)[i] for i in settled])
    theirs = statistics.fmean([deviation(legacy)[i] for i in settled])
    assert mine < theirs + 8
    assert abs(mapper.precision.offset[0]) <= PRECISION.limit*mapper.span + 1e-6


def test_repeated_slow_fast_cycles_do_not_accumulate_displacement():
    """The offset used to be an integral of the frame-to-frame step, so slow and
    fast work in alternation random-walked the cursor away from the map."""
    frames = stream(cycles(), 24.0, seed=9)
    mapper = CursorMapper(BOUNDS)
    rows = pointer(mapper, frames)
    legacy = pointer(Legacy(), frames)
    limit = PRECISION.limit*mapper.span
    # The clutch's own contribution is bounded frame by frame ...
    assert max(abs(row[2][0]-row[3][0]) for row in rows) < limit + max(
        abs(row[2][0]-row[3][0]) for row in legacy) + 5
    # ... and does not grow over twelve cycles of the same movement.
    early = statistics.fmean(deviation(rows[:60]))
    late = statistics.fmean(deviation(rows[-60:]))
    assert late < early + 10


def test_the_map_stays_memoryless_however_the_hand_arrives():
    """The same hand position must mean the same screen position. Before this,
    four approaches to one point settled 178 px apart; the pointer had stopped
    being a map and become a function of its own history."""
    settled = []
    for start, speed in ((.50, .04), (.30, .90), (.70, .04), (.90, .90),
                         (.58, .02), (.62, .02)):
        travel = abs(.60-start)/speed
        def path(t, start=start, travel=travel):
            return ((start+(.60-start)*t/travel, .5) if t < travel else (.60, .5))
        rows = pointer(CursorMapper(BOUNDS), stream(path, travel+2.0, seed=11))
        settled.append(statistics.fmean([r[2][0] for r in rows[-15:]]))
    assert max(settled)-min(settled) < 2*PRECISION.limit*BOUNDS[2] + 5


# --- stationary jitter ---------------------------------------------------------

@pytest.mark.parametrize('noise', (.002, .003, .004))
def test_a_still_hand_does_not_vibrate_the_cursor(noise):
    frames = stream(still(), 8.0, seed=1, noise=noise)
    new, legacy = both(frames)
    fresh, old = wobble(new), wobble(legacy)
    assert fresh[0] < old[0]                  # peak-to-peak
    assert fresh[1] < old[1]                  # RMS
    assert fresh[2] <= old[2] + .5            # largest single-frame step
    assert fresh[0] < 16                      # and small enough not to be seen


def test_stationary_jitter_is_not_a_slow_wander():
    """High-frequency jitter averages out visually; a random walk does not, and
    integrating the noise produced exactly one. Split the trace and compare the
    means: they should sit on top of each other."""
    rows = pointer(CursorMapper(BOUNDS), stream(still(), 20.0, seed=13))
    half = len(rows)//2
    first = statistics.fmean([r[2][0] for r in rows[:half]])
    second = statistics.fmean([r[2][0] for r in rows[half:]])
    assert abs(first-second) < 6


def test_landmark_spikes_do_not_throw_the_cursor():
    frames = stream(still(), 8.0, seed=17, spikes=.04)
    new, legacy = both(frames)
    assert wobble(new)[2] <= wobble(legacy)[2] + .5


# --- precision mode ------------------------------------------------------------

def test_settling_onto_a_target_engages_precision():
    mapper = CursorMapper(BOUNDS)
    rows = pointer(mapper, stream(acquire(), 3.0, seed=19))
    assert mapper.precision.active


def test_precision_makes_the_same_small_movement_move_the_cursor_less():
    """What the feature is for. Measured after the mode has engaged, so this is
    the assistance and not the approach."""
    frames = stream(steady(.03, start=.45), 2.0, seed=21)
    new, legacy = both(frames)
    # Over a correction the size of the assisted range, which is what the
    # clutch is for; a longer movement runs past the cap and tracks the map
    # one-to-one again, by design.
    reach = PRECISION.limit*BOUNDS[2]/(1-PRECISION.gain)
    def travelled(rows):
        start = next(r for r in rows if r[0] > 1.0)
        end = next((r for r in rows if r[0] > 1.0
                    and math.dist(r[3], start[3]) >= reach), rows[-1])
        return math.dist(start[2], end[2])
    assert travelled(new) < travelled(legacy)*.8


def test_precision_does_not_chatter_around_its_thresholds():
    """A single threshold that the noise floor straddles would flip the control
    mode frame to frame. Swept right through the hysteresis band."""
    for speed in (.06, .08, .10, .12, .14, .16, .18, .20):
        mapper = CursorMapper(BOUNDS)
        flips = 0
        previous = None
        for row in pointer(mapper, stream(steady(speed, start=.2), 8.0, seed=23)):
            if previous is not None and mapper.precision.active != previous:
                flips += 1
            previous = mapper.precision.active
        assert flips <= 2, f'{speed} u/s flipped {flips} times'


def test_entering_and_leaving_precision_is_continuous():
    """Neither transition may show as a jump. The largest single-frame step over
    a trace that crosses the boundary repeatedly is the whole test."""
    mapper = CursorMapper(BOUNDS)
    rows = pointer(mapper, stream(cycles(period=1.2), 16.0, seed=27))
    legacy = pointer(Legacy(), stream(cycles(period=1.2), 16.0, seed=27))
    assert wobble(rows)[2] <= wobble(legacy)[2] + 1


def test_pushing_into_the_margin_still_reaches_the_screen_edge():
    """An offset frozen against a saturated axis would put the last strip of the
    desktop permanently out of reach."""
    mapper = CursorMapper(BOUNDS)
    rows = pointer(mapper, stream(steady(.02, start=.45), 6.0, seed=29))
    now = rows[-1][0]
    for _ in range(60):
        now += 1/FPS
        out = mapper.update((.99, .5), S, 1/FPS, now)
    assert abs(mapper.precision.offset[0]) < 5
    assert out[0] == pytest.approx(BOUNDS[2]-1, abs=3)


# --- the speed estimate --------------------------------------------------------

def test_the_intent_speed_is_robust_to_landmark_noise():
    """It is only ever compared against a threshold, so it can afford to be
    lower bandwidth than the pointer -- and has to be, because a derivative of
    noisy landmarks reads a still hand as moving."""
    estimator = VelocityEstimator()
    frames = stream(still(), 6.0, seed=31)
    seen = [estimator.update(point, now) for now, point in frames]
    assert max(seen[10:]) < PRECISION.enter_speed
    assert statistics.fmean(seen[10:]) < PRECISION.enter_speed/2


@pytest.mark.parametrize('speed', (.05, .3, 1.0, 2.0))
def test_the_intent_speed_is_accurate_on_a_steady_hand(speed):
    estimator = VelocityEstimator()
    frames = stream(steady(speed, start=.1), 3.0, seed=33)
    seen = [estimator.update(point, now) for now, point in frames]
    assert statistics.fmean(seen[15:]) == pytest.approx(speed, rel=.2)


def test_uneven_frame_timing_does_not_bias_the_speed():
    """Regression against the least-squares slope being replaced by something
    that assumes a fixed frame interval."""
    estimator = VelocityEstimator()
    frames = stream(steady(.5, start=.1), 3.0, seed=35, timing=.012)
    seen = [estimator.update(point, now) for now, point in frames]
    assert statistics.fmean(seen[15:]) == pytest.approx(.5, rel=.2)


def test_a_dropped_stretch_of_frames_restarts_the_window():
    estimator = VelocityEstimator()
    estimator.update((.2, .5), 1.0)
    estimator.update((.8, .5), 9.0)
    assert estimator.raw == 0.0
    assert estimator.speed == 0.0


def test_the_clutch_carries_no_state_across_a_handover():
    """A drag or a throw owns the pointer for a while and the absolute target
    moves a long way meanwhile; resuming must not read that as hand movement."""
    mapper = CursorMapper(BOUNDS)
    rows = pointer(mapper, stream(still(at=(.3, .5)), 2.0, seed=37))
    now = rows[-1][0] + .8                     # something else had the pointer
    mapper.update((.75, .5), S, 1/FPS, now)
    assert not mapper.precision.active
    assert mapper.precision.offset == (0.0, 0.0)
