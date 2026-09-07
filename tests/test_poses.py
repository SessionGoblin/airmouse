import math
import random

import pytest

from airmouse import poses
from airmouse.config import Settings
from airmouse.gestures import Features, GestureMachine, State


def hand(seed=1):
    """21 arbitrary but fixed landmarks. The anatomy is irrelevant to the
    invariance properties under test; only the wrist and middle-MCP matter,
    because those two define the normalization frame."""
    rng = random.Random(seed)
    points = [(rng.uniform(.3, .7), rng.uniform(.2, .8), 0.0) for _ in range(21)]
    points[poses.WRIST] = (.5, .8, 0.0)
    points[poses.MIDDLE_MCP] = (.5, .5, 0.0)
    return points


def transform(points, angle=0.0, scale=1.0, dx=0.0, dy=0.0):
    cos, sin = math.cos(angle), math.sin(angle)
    return [((x*cos - y*sin)*scale + dx, (x*sin + y*cos)*scale + dy, z)
            for x, y, z in points]


def test_pose_is_invariant_to_position_scale_and_tilt():
    """The same shape held elsewhere in frame, nearer the camera, and tilted
    must land on one template rather than needing one template per pose."""
    base, _ = poses.normalize(hand(), 1.0)
    moved, _ = poses.normalize(transform(hand(), angle=.45, scale=1.8, dx=.12, dy=-.07), 1.0)
    assert poses.distance(base, moved) < 1e-9


def test_orientation_tracks_tilt_so_flipped_poses_stay_separable():
    """Thumbs-up and thumbs-down are the same shape; only orientation splits
    them, so rotation normalization must not throw the angle away."""
    _, upright = poses.normalize(hand(), 1.0)
    _, rolled = poses.normalize(transform(hand(), angle=.6), 1.0)
    assert abs(poses.angle_delta(rolled, upright) - .6) < 1e-9
    _, flipped = poses.normalize(transform(hand(), angle=math.pi), 1.0)
    assert abs(abs(poses.angle_delta(flipped, upright)) - math.pi) < 1e-9


def test_wrist_anchors_the_frame_and_span_becomes_the_unit():
    pose, _ = poses.normalize(hand(), 1.0)
    assert pose[poses.WRIST] == pytest.approx((0.0, 0.0), abs=1e-9)
    # The wrist -> middle-MCP span is rotated onto "up" and scaled to 1.
    assert pose[poses.MIDDLE_MCP] == pytest.approx((0.0, -1.0), abs=1e-9)


def test_degenerate_hands_do_not_produce_a_pose():
    assert poses.normalize(None, 1.0) == (None, None)
    assert poses.normalize([(0., 0., 0.)]*5, 1.0) == (None, None)
    flat = [(.5, .5, 0.)]*21                    # wrist and MCP coincident
    assert poses.normalize(flat, 1.0) == (None, None)


def test_threshold_follows_recorded_spread():
    """A shaky recording earns a roomier radius than a steady one, instead of
    every template sharing one guessed constant."""
    steady = [poses.normalize(transform(hand(), angle=a), 1.0)[0] for a in (0, .002, -.002)]
    shaky = [poses.normalize(hand(s), 1.0)[0] for s in (1, 2, 3)]
    assert poses.build('steady', steady).threshold < poses.build('shaky', shaky).threshold


def test_orientation_averages_across_the_pi_wrap():
    """Samples straddling +/-pi must not average to zero and point at the
    opposite tilt from the one that was recorded."""
    template = poses.build('flip', [poses.normalize(hand(), 1.0)[0]]*2,
                           orientations=[math.pi-.05, -math.pi+.05])
    assert abs(abs(template.orientation) - math.pi) < .06


def test_match_picks_the_closest_accepting_template():
    near, _ = poses.normalize(hand(1), 1.0)
    far, _ = poses.normalize(hand(7), 1.0)
    library = poses.Library([
        poses.Template(name='near', pose=near, threshold=.5),
        poses.Template(name='far', pose=far, threshold=.5),
    ])
    assert library.match(near, 0.0).name == 'near'


def test_match_rejects_a_pose_outside_every_radius():
    near, _ = poses.normalize(hand(1), 1.0)
    far, _ = poses.normalize(hand(7), 1.0)
    library = poses.Library([poses.Template(name='near', pose=near, threshold=.01)])
    assert library.match(far, 0.0) is None


def test_orientation_bound_rejects_the_same_shape_at_the_wrong_tilt():
    pose, angle = poses.normalize(hand(), 1.0)
    template = poses.Template(name='up', pose=pose, threshold=.5,
                              orientation=angle, tolerance=math.radians(30))
    assert template.matches(pose, angle) is not None
    assert template.matches(pose, angle + math.radians(90)) is None
    # Without an orientation the same template accepts any tilt.
    assert poses.Template(name='any', pose=pose, threshold=.5).matches(pose, angle + 2) is not None


def test_conflict_flags_a_near_duplicate_that_could_never_win():
    pose, _ = poses.normalize(hand(), 1.0)
    library = poses.Library([poses.Template(name='first', pose=pose, threshold=.2)])
    twin = poses.Template(name='second', pose=pose, threshold=.2)
    assert library.conflict(twin).name == 'first'
    distinct = poses.Template(name='other', pose=poses.normalize(hand(9), 1.0)[0], threshold=.2)
    assert library.conflict(distinct) is None


def test_library_round_trips_through_disk(tmp_path):
    pose, angle = poses.normalize(hand(), 1.0)
    library = poses.Library([poses.Template(name='fist', action='shell', argument='true',
                                            pose=pose, threshold=.2, orientation=angle)])
    path = tmp_path / 'gestures.json'
    library.save(path)
    loaded = poses.Library.load(path)
    assert [t.name for t in loaded.templates] == ['fist']
    assert loaded.templates[0].argument == 'true'
    assert poses.distance(loaded.templates[0].pose, pose) < 1e-9


def test_library_load_survives_a_corrupt_or_missing_file(tmp_path):
    assert poses.Library.load(tmp_path / 'absent.json').templates == []
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json')
    assert poses.Library.load(bad).templates == []
    partial = tmp_path / 'partial.json'
    partial.write_text('[{"name": "short", "pose": [[0, 0]]}, {"nonsense": 1}]')
    assert poses.Library.load(partial).templates == []   # wrong landmark count dropped


# --- state machine integration -------------------------------------------------

S = Settings()


def machine_with(pose, **binding):
    template = poses.Template(name='wave', pose=pose, threshold=.5, **binding)
    return GestureMachine(poses.Library([template])), template


def custom_features(pose, angle=0.0):
    """Neutral pinch distances, so anything that fires came from the pose."""
    return Features((.5, .5), .8, .8, False, pose, angle)


def test_custom_pose_fires_once_after_its_dwell():
    pose, _ = poses.normalize(hand(), 1.0)
    machine, template = machine_with(pose)
    machine.armed = True
    f = custom_features(pose)
    assert machine.step(f, 1.0, S, True) == []                      # confirming
    assert machine.step(f, 1.0 + S.custom_dwell/2, S, True) == []    # still confirming
    assert machine.step(f, 1.0 + S.custom_dwell + .01, S, True) == [('custom', template)]
    assert machine.state is State.CUSTOM
    # Latched: holding the pose must not re-run the bound action every frame.
    assert machine.step(f, 1.5, S, True) == []
    assert machine.step(f, 2.0, S, True) == []


def test_leaving_and_re_forming_the_pose_fires_again():
    pose, _ = poses.normalize(hand(), 1.0)
    machine, template = machine_with(pose)
    machine.armed = True
    f = custom_features(pose)
    machine.step(f, 1.0, S, True)
    assert machine.step(f, 1.0 + S.custom_dwell + .01, S, True) == [('custom', template)]
    machine.step(custom_features(poses.normalize(hand(9), 1.0)[0]), 2.0, S, True)   # released
    machine.step(f, 3.0, S, True)
    assert machine.step(f, 3.0 + S.custom_dwell + .01, S, True) == [('custom', template)]


def test_custom_poses_can_be_switched_off():
    pose, _ = poses.normalize(hand(), 1.0)
    machine, _ = machine_with(pose)
    machine.armed = True
    off = Settings(custom=False)
    f = custom_features(pose)
    machine.step(f, 1.0, off, True)
    # Falls through to pointing rather than firing.
    assert machine.step(f, 1.0 + off.custom_dwell + .01, off, True) == [('move', (.5, .5))]


def test_a_drag_in_progress_outranks_a_matching_pose():
    """Templates are matched before the pinches, but never while the button is
    already down, or a pose crossed mid-drag would drop the window."""
    pose, _ = poses.normalize(hand(), 1.0)
    machine, _ = machine_with(pose)
    machine.armed = True
    machine.down, machine.pressed_at = True, 0.0
    actions = machine.step(Features((.6, .4), .1, .8, False, pose, 0.0), 1.0, S, True)
    assert actions == [('move', (.6, .4))]
    assert machine.state is State.DRAGGING


def test_reset_keeps_the_template_library():
    """reset() re-runs __init__; the library must survive a lost hand."""
    pose, _ = poses.normalize(hand(), 1.0)
    machine, _ = machine_with(pose)
    machine.reset()
    assert machine.library is not None
    machine.armed = True
    f = custom_features(pose)
    machine.step(f, 1.0, S, True)
    assert machine.step(f, 1.0 + S.custom_dwell + .01, S, True)[0][0] == 'custom'


def test_threshold_floor_accepts_the_same_pose_re_formed():
    """The regression that made every recorded gesture dead on arrival: the
    accept radius was set below the frame-to-frame variation of a single pose,
    so nothing ever matched its own template."""
    import random
    base = hand(4)
    samples = []
    for i in range(30):                     # a steady hold, as the recorder sees it
        rng = random.Random(i)
        jittered = [(x+rng.gauss(0, .002), y+rng.gauss(0, .002), z) for x, y, z in base]
        samples.append(poses.normalize(jittered, 1.0)[0])
    template = poses.build('held', samples)
    # The same pose re-formed later, with realistic landmark noise.
    misses = 0
    for i in range(40):
        rng = random.Random(1000+i)
        again = [(x+rng.gauss(0, .012), y+rng.gauss(0, .012), z) for x, y, z in base]
        if template.matches(poses.normalize(again, 1.0)[0], None) is None:
            misses += 1
    assert misses == 0, f'{misses}/40 re-formed poses fell outside the radius'


def test_distinct_shapes_stay_far_outside_the_radius():
    """The floor may be generous, but different hand shapes sit far enough
    apart that it does not cause collisions."""
    a, _ = poses.normalize(hand(1), 1.0)
    template = poses.Template(name='a', pose=a, threshold=poses.MAX_THRESHOLD)
    for seed in range(2, 12):
        other, _ = poses.normalize(hand(seed), 1.0)
        assert template.matches(other, None) is None


def test_old_tight_templates_are_widened_on_load(tmp_path):
    """Gestures recorded before the floor was corrected must start working
    rather than staying silently dead."""
    pose, _ = poses.normalize(hand(), 1.0)
    path = tmp_path/'gestures.json'
    poses.Library([poses.Template(name='old', pose=pose, threshold=.06)]).save(path)
    assert poses.Library.load(path).templates[0].threshold == poses.MIN_THRESHOLD


def test_conflict_uses_the_accept_radius_not_a_constant():
    """Two templates overlap when their centres are closer than a radius, so
    the check scales with how wide the templates actually are."""
    a, _ = poses.normalize(hand(1), 1.0)
    b, _ = poses.normalize(hand(2), 1.0)
    gap = poses.distance(a, b)
    narrow = poses.Library([poses.Template(name='a', pose=a, threshold=gap*.5)])
    assert narrow.conflict(poses.Template(name='b', pose=b, threshold=gap*.5)) is None
    wide = poses.Library([poses.Template(name='a', pose=a, threshold=gap*1.2)])
    assert wide.conflict(poses.Template(name='b', pose=b, threshold=gap*1.2)).name == 'a'


def test_nearest_reports_distance_even_when_nothing_matches():
    a, _ = poses.normalize(hand(1), 1.0)
    b, _ = poses.normalize(hand(7), 1.0)
    library = poses.Library([poses.Template(name='a', pose=a, threshold=.01)])
    assert library.match(b, None) is None
    template, gap = library.nearest(b)
    assert template.name == 'a' and gap > .01
    assert poses.Library().nearest(a) == (None, None)
