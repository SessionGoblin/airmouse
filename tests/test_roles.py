from airmouse.roles import Hand, RoleAssigner, POINTER, MODIFIER, LOST, by_role


def hand(x, y=.5):
    """A hand whose wrist sits at (x, y); only the anchor matters here."""
    return Hand(points=[(x, y, 0.)] + [(x, y, 0.)]*20)


def roles(hands):
    return [h.role for h in hands]


def test_a_lone_hand_drives_the_cursor():
    """Assigning a single hand to the modifier would leave no pointer at all."""
    a = RoleAssigner()
    assert roles(a.assign([hand(.3)], 0.0)) == [POINTER]


def test_two_fresh_hands_seed_by_side_of_frame():
    """Seeded from position, not the handedness label: the frame is mirrored
    before inference, so the right hand is reliably on the right of the image."""
    a = RoleAssigner(pointer_side='right')
    assert roles(a.assign([hand(.2), hand(.8)], 0.0)) == [MODIFIER, POINTER]
    b = RoleAssigner(pointer_side='left')
    assert roles(b.assign([hand(.2), hand(.8)], 0.0)) == [POINTER, MODIFIER]


def test_roles_follow_the_hands_rather_than_the_detection_order():
    """The model returns hands in arbitrary order; roles must not ride on it."""
    a = RoleAssigner()
    a.assign([hand(.2), hand(.8)], 0.0)                 # left=modifier, right=pointer
    swapped = a.assign([hand(.8), hand(.2)], 0.1)       # same hands, reversed order
    assert roles(swapped) == [POINTER, MODIFIER]


def test_hands_that_cross_keep_their_roles_through_the_crossing():
    """The failure that makes multi-hand unusable: roles trading places mid
    gesture, handing the cursor to the other hand.

    Matching on predicted position rather than last position is what carries
    the roles through: each hand keeps moving the way it was, so it stays
    nearest its own prediction even once it is past the other hand. The hands
    move at different speeds, as real ones do -- two hands crossing at exactly
    mirrored speeds are momentarily coincident and genuinely unresolvable from
    geometry alone.
    """
    a = RoleAssigner()
    a.assign([hand(.2), hand(.8)], 0.0)
    now = 0.0
    for step, (left, right) in enumerate([(.30,.74), (.40,.68), (.50,.62), (.60,.56),
                                          (.70,.50), (.80,.44)]):
        now += .033
        got = a.assign([hand(left), hand(right)], now)
        assert got[0].role == MODIFIER, f'modifier lost its role at step {step}'
        assert got[1].role == POINTER, f'pointer lost its role at step {step}'


def test_a_deliberate_swap_still_takes_effect():
    """Stickiness must not become a refusal to ever reassign."""
    a = RoleAssigner()
    a.assign([hand(.2), hand(.8)], 0.0)
    # The pointer leaves and comes back on the far side, well past the margin.
    got = a.assign([hand(.85)], .05)
    assert got[0].role == POINTER
    got = a.assign([hand(.15), hand(.85)], .1)
    assert roles(got) == [MODIFIER, POINTER]


def test_a_brief_dropout_does_not_reshuffle_roles():
    a = RoleAssigner()
    a.assign([hand(.2), hand(.8)], 0.0)
    a.assign([hand(.8)], .05)                    # modifier momentarily undetected
    back = a.assign([hand(.21), hand(.79)], .1)
    assert roles(back) == [MODIFIER, POINTER]


def test_a_role_is_released_once_its_hand_is_gone_for_good():
    a = RoleAssigner()
    a.assign([hand(.2), hand(.8)], 0.0)
    a.assign([], 0.1)
    # After the timeout a single hand on the old modifier side takes the cursor,
    # rather than staying a modifier with nothing pointing.
    late = a.assign([hand(.2)], 0.1 + LOST + .1)
    assert late[0].role == POINTER


def test_a_returning_hand_resumes_its_own_role_within_the_timeout():
    a = RoleAssigner()
    a.assign([hand(.2), hand(.8)], 0.0)
    back = a.assign([hand(.22)], LOST/2)
    assert back[0].role == MODIFIER          # it is the modifier's hand, not a new pointer


def test_close_hands_hold_their_roles_under_jitter():
    """Two hands held near each other must not trade roles on landmark noise."""
    import random
    a = RoleAssigner()
    settled = roles(a.assign([hand(.45), hand(.55)], 0.0))
    rng = random.Random(0)
    for i in range(60):
        got = a.assign([hand(.45 + rng.gauss(0, .004)), hand(.55 + rng.gauss(0, .004))],
                       .033 * (i+1))
        assert roles(got) == settled, f'roles swapped on noise at frame {i}'


def test_one_remembered_role_places_both_hands():
    """A second hand appearing beside a tracked one must not reseed by side and
    steal the cursor from the hand already holding it."""
    a = RoleAssigner()
    a.assign([hand(.2)], 0.0)                    # lone hand takes the pointer
    got = a.assign([hand(.21), hand(.8)], .033)
    assert roles(got) == [POINTER, MODIFIER]


def test_hands_without_landmarks_are_ignored():
    a = RoleAssigner()
    got = a.assign([Hand(points=[]), hand(.5)], 0.0)
    assert got[0].role is None and got[1].role == POINTER


def test_a_third_hand_gets_no_role():
    """Two roles exist; a spurious third detection must not take one."""
    a = RoleAssigner()
    got = a.assign([hand(.2), hand(.8), hand(.5)], 0.0)
    assert sorted(r for r in roles(got) if r) == [MODIFIER, POINTER]
    assert roles(got).count(None) == 1


def test_by_role_finds_the_assigned_hand():
    a = RoleAssigner()
    hands = a.assign([hand(.2), hand(.8)], 0.0)
    assert by_role(hands, POINTER).anchor[0] == .8
    assert by_role(hands, MODIFIER).anchor[0] == .2
    assert by_role([], POINTER) is None


def test_reset_clears_remembered_positions():
    a = RoleAssigner()
    a.assign([hand(.2), hand(.8)], 0.0)
    a.reset()
    assert roles(a.assign([hand(.8), hand(.2)], .05)) == [POINTER, MODIFIER]
