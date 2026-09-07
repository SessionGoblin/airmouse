"""Releasing a drag mid-motion coasts the pointer with the button still held."""
import math

import pytest

from airmouse import controller as C
from airmouse.config import Settings
from airmouse.controller import Controller
from airmouse.gestures import Features, State

BOUNDS = (0, 0, 1920, 1080)


class Backend:
    """Records button and motion events in order."""
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


def drag(c, speed=.02, frames=8, start=1.0, y=.5, fps=30, x=.22):
    """Pinch and sweep at `speed` in normalized units per frame.

    Starts near the left of the active region and stops well short of the right
    edge: a sweep that ends at the screen boundary leaves the coast no room and
    it stops on its first step.
    """
    now = start
    c.process(Features((x, y), .1, .8, False), now)          # pinch down
    for _ in range(frames):
        now += 1/fps
        x += speed
        c.process(Features((x, y), .1, .8, False), now)
    return now, x, y


def release(c, now, x, y, fps=30):
    return c.process(Features((x, y), .8, .8, False), now + 1/fps)


def coast(c, now, point, frames=40, fps=30):
    """Run frames after the release with the hand open and still.

    The hand stays where it was released. Teleporting it would trip the
    discontinuity guard, which treats a quarter-frame jump as lost tracking and
    cancels the throw -- correctly, but that is not what this is testing.
    """
    states = []
    for i in range(frames):
        now += 1/fps
        states.append(c.process(Features(point, .8, .8, False), now))
    return states


def test_a_fast_release_keeps_the_button_down_and_coasts():
    """The whole point: a compositor stops moving a window the moment the
    button comes up, so the throw has to hold it through the coast."""
    c, backend = build()
    now, x, y = drag(c)
    release(c, now, x, y)
    assert c.throw is not None
    assert backend.events.count('up') == 0          # still held
    states = coast(c, now + .033, (x, y), frames=6)
    assert all(s == State.THROWN.value for s in states)
    assert backend.events.count('up') == 0


def test_the_coast_ends_and_releases():
    c, backend = build()
    now, x, y = drag(c)
    release(c, now, x, y)
    coast(c, now + .033, (x, y), frames=60)
    assert c.throw is None
    assert backend.events.count('up') == 1          # released exactly once


def test_the_pointer_keeps_travelling_after_the_hand_stops():
    c, backend = build()
    now, x, y = drag(c)
    release(c, now, x, y)
    landed = c.target[0]
    furthest = landed
    at = now + .033
    while c.throw is not None:
        at += .033
        c.process(Features((x, y), .8, .8, False), at)
        furthest = max(furthest, c.target[0])
    assert furthest > landed + 200                  # carried well past the release
    # Once the coast is over, pointing resumes and the cursor returns to the hand.
    coast(c, at, (x, y), frames=20)
    assert c.target[0] < furthest


def test_velocity_decays_rather_than_running_at_constant_speed():
    c, _ = build()
    now, x, y = drag(c)
    release(c, now, x, y)
    steps = []
    previous = c.target[0]
    for i in range(8):
        c.process(Features((x, y), .8, .8, False), now + .033*(i+2))
        if c.throw is None:
            break
        steps.append(c.target[0] - previous)
        previous = c.target[0]
    assert len(steps) > 3
    assert steps[-1] < steps[0]                     # slowing down


def test_a_slow_release_is_just_a_release():
    c, backend = build()
    now, x, y = drag(c, .002)                       # barely moving
    release(c, now, x, y)
    assert c.throw is None
    assert backend.events[-1] == 'up'


def test_a_click_never_throws():
    """No drag started, so there is nothing to fling."""
    c, backend = build()
    c.process(Features((.5, .5), .1, .8, False), 1.0)
    c.process(Features((.5, .5), .8, .8, False), 1.05)
    assert c.throw is None
    assert not c.drag_started


def test_a_lost_hand_mid_drag_releases_instead_of_throwing():
    """Tracking failure is not intent: a hand that vanishes mid-sweep must drop
    the window where it is, not hurl it across the desktop."""
    c, backend = build()
    now, x, y = drag(c)
    c.process(None, now + .033)
    assert c.throw is None
    assert backend.events.count('up') == 1


def test_pause_cancels_a_throw_in_flight():
    """F12 must not leave the button held by a coast nobody is watching."""
    c, backend = build()
    now, x, y = drag(c)
    release(c, now, x, y)
    assert c.throw is not None
    c.pause()
    assert c.throw is None
    assert backend.events[-1] == 'up'


def test_re_pinching_catches_the_throw():
    c, backend = build()
    now, x, y = drag(c)
    release(c, now, x, y)
    c.process(Features((x, y), .1, .8, False), now + .066)      # pinch again
    assert c.throw is None
    assert backend.events.count('up') == 1


def test_drawing_cancels_a_throw():
    c, backend = build()
    c.stroke_library = None
    now, x, y = drag(c)
    release(c, now, x, y)
    state = c.process(Features((x, y), .8, .8, False), now + .066,
                      Features((.2, .5), .1, .8, False))         # modifier pinch
    assert state == State.DRAWING.value
    assert c.throw is None


def test_a_throw_stops_at_the_desktop_edge():
    """Rather than grinding along it for the rest of the coast."""
    c, backend = build()
    now, x, y = drag(c, .06, frames=16, x=.3)       # aimed hard at the right edge
    release(c, now, x, y)
    coast(c, now + .033, (x, y), frames=60)
    assert c.throw is None
    assert c.target[0] <= BOUNDS[2] - 1
    assert backend.events.count('up') == 1          # released, not left held at the edge


def test_the_deadline_caps_how_long_the_button_can_stay_down():
    c, backend = build()
    now, x, y = drag(c)
    release(c, now, x, y)
    held = 0
    for i in range(200):
        c.process(Features((x, y), .8, .8, False), now + .033*(i+2))
        if c.throw is None:
            break
        held += 1
    assert held * .033 < C.FLING_DEADLINE + .1
    assert backend.events.count('up') == 1


def test_fling_can_be_switched_off():
    c, backend = build(fling=False)
    now, x, y = drag(c)
    release(c, now, x, y)
    assert c.throw is None
    assert backend.events[-1] == 'up'


# --- velocity estimation -------------------------------------------------------

def test_velocity_ignores_the_frames_just_before_release():
    """Opening a pinch drags the index fingertip sideways for a frame or two;
    including that curves every throw toward wherever the thumb went."""
    c, _ = build()
    now = 10.0
    for i in range(10):                             # steady rightward sweep
        c.trail.append((now - TRAIL_STEP*(9-i), (100 + i*40, 500)))
    # Two frames of sharp downward jerk, as the hand opens.
    c.trail.append((now - .02, (460, 700)))
    c.trail.append((now - .01, (460, 900)))
    velocity = c.fling_velocity(now)
    assert velocity[0] > 0
    assert abs(velocity[1]) < 50                    # the jerk was excluded


TRAIL_STEP = .018


def test_too_few_samples_gives_no_velocity():
    c, _ = build()
    assert c.fling_velocity(10.0) is None
    c.trail.append((9.9, (100, 100)))
    assert c.fling_velocity(10.0) is None


def test_a_stale_trail_gives_no_velocity():
    """Samples older than the window must not resurrect an ancient sweep."""
    c, _ = build()
    for i in range(5):
        c.trail.append((5.0 + i*.02, (100 + i*50, 500)))
    assert c.fling_velocity(10.0) is None
