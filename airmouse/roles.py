"""Assign detected hands to pointer / modifier roles and hold them steady.

Pure geometry over wrist positions, so role stability is testable without a
camera or the model. Deliberately does not use the model's handedness label:
that classifier flickers exactly when two hands are close or overlapping, which
is when a role swap is most damaging -- it would drop a drag and hand the cursor
to the other hand mid-gesture. Continuity of position is the reliable signal.
"""
from dataclasses import dataclass, field

POINTER, MODIFIER = 'pointer', 'modifier'
ROLES = (POINTER, MODIFIER)

# A role keeps its slot this long after its hand leaves, so a brief detection
# dropout does not reshuffle roles and, with them, which hand owns the cursor.
LOST = .4
# Roles are matched against where each hand is predicted to be, not where it
# was. Hands reaching across each other otherwise trade roles at the crossing:
# each ends up nearest the *other* hand's last position, which hands the cursor
# over mid-gesture. Prediction is not extrapolated beyond this horizon, since a
# stale velocity is worse than none.
PREDICT_MAX = .1
# Remembered positions are always the previous frame's, so nearest-assignment
# is itself the continuity test: across a crossing each hand stays closer to
# where it just was than to where the other hand just was. No extra hysteresis
# constant is needed, and a margin biased toward detection order -- which the
# model does not keep stable -- would be worse than none.


@dataclass
class Hand:
    """One detected hand and what the pipeline made of it."""
    points: list
    label: str = ''            # the model's handedness, kept for display only
    score: float = 0.0
    role: str = None
    features: object = None

    @property
    def anchor(self):
        return self.points[0][:2] if self.points else None


def _distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** .5


@dataclass
class RoleAssigner:
    """Sticky pointer / modifier assignment across frames."""

    # Which side of the mirrored preview seeds the pointer. Position, not
    # handedness: the frame is mirrored before inference, so your right hand is
    # reliably on the right of the image whatever the classifier says.
    pointer_side: str = 'right'
    position: dict = field(default_factory=lambda: {POINTER: None, MODIFIER: None})
    velocity: dict = field(default_factory=lambda: {POINTER: None, MODIFIER: None})
    seen: dict = field(default_factory=lambda: {POINTER: None, MODIFIER: None})

    def reset(self):
        for role in ROLES:
            self.position[role] = None
            self.velocity[role] = None
            self.seen[role] = None

    def _expire(self, now):
        for role in ROLES:
            if self.seen[role] is not None and now - self.seen[role] > LOST:
                self.position[role] = None
                self.velocity[role] = None
                self.seen[role] = None

    def _predict(self, role, now):
        """Where this role's hand should be by now, given how it was moving."""
        anchor = self.position[role]
        if anchor is None:
            return None
        speed = self.velocity[role]
        if speed is None or self.seen[role] is None:
            return anchor
        step = min(now - self.seen[role], PREDICT_MAX)
        return (anchor[0] + speed[0]*step, anchor[1] + speed[1]*step)

    def _remember(self, role, anchor, now):
        previous, seen = self.position[role], self.seen[role]
        if previous is not None and seen is not None and 0 < now-seen < LOST:
            step = now - seen
            fresh = ((anchor[0]-previous[0])/step, (anchor[1]-previous[1])/step)
            old = self.velocity[role]
            # Blend, so one noisy frame cannot throw the next prediction.
            self.velocity[role] = fresh if old is None else tuple(
                (a+b)/2 for a, b in zip(old, fresh))
        elif previous is None:
            self.velocity[role] = None
        self.position[role] = anchor
        self.seen[role] = now

    def _seed_pointer(self, anchors):
        """Which of two fresh hands drives the cursor."""
        rightmost = max(range(len(anchors)), key=lambda i: anchors[i][0])
        leftmost = min(range(len(anchors)), key=lambda i: anchors[i][0])
        return rightmost if self.pointer_side == 'right' else leftmost

    def assign(self, hands, now):
        """Set .role on each hand. Returns the same list for convenience."""
        for hand in hands:
            hand.role = None
        self._expire(now)
        anchors = [h.anchor for h in hands if h.anchor is not None]
        usable = [h for h in hands if h.anchor is not None]
        if not usable:
            return hands
        if len(usable) == 1:
            self._assign_single(usable[0], now)
            return hands
        # Exactly two pairings exist for two hands, so the cheaper one is found
        # by comparing them rather than by a general assignment algorithm.
        first, second = usable[0], usable[1]
        known = [role for role in ROLES if self.position[role] is not None]
        if len(known) == 2:
            straight = self._cost(first, POINTER, now) + self._cost(second, MODIFIER, now)
            crossed = self._cost(first, MODIFIER, now) + self._cost(second, POINTER, now)
            pairing = (POINTER, MODIFIER) if straight <= crossed else (MODIFIER, POINTER)
        elif known:
            # Only one role is remembered: the nearer hand continues it, and the
            # other takes the free role.
            role = known[0]
            free = MODIFIER if role == POINTER else POINTER
            pairing = ((role, free) if self._cost(first, role, now) <= self._cost(second, role, now)
                       else (free, role))
        else:
            index = self._seed_pointer([first.anchor, second.anchor])
            pairing = (POINTER, MODIFIER) if index == 0 else (MODIFIER, POINTER)
        first.role, second.role = pairing
        for hand in usable[2:]:
            hand.role = None
        for hand in (first, second):
            self._remember(hand.role, hand.anchor, now)
        return hands

    def _cost(self, hand, role, now):
        """Distance from a hand to where that role is predicted to be."""
        predicted = self._predict(role, now)
        return None if predicted is None else _distance(hand.anchor, predicted)

    def _assign_single(self, hand, now):
        """One hand: continue the role it belongs to rather than defaulting."""
        costs = {role: self._cost(hand, role, now) for role in ROLES}
        known = {role: c for role, c in costs.items() if c is not None}
        if known:
            role = min(known, key=known.get)
        else:
            # Nothing remembered: a lone hand drives the cursor. Picking the
            # modifier here would leave the user with no pointer at all.
            role = POINTER
        hand.role = role
        self._remember(role, hand.anchor, now)


def by_role(hands, role):
    return next((h for h in hands if h.role == role), None)
