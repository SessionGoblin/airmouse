"""Normalized hand point -> desktop pixels, and every pointer filter.

One owner for pointer filtering, so each interaction state gets the control
characteristics it wants without a second copy of the maths living elsewhere.

The absolute map is the authority and it is deliberately *memoryless*: a hand
position maps to one screen position, whatever route the hand took to get
there. That property is what makes the pointer learnable, and it is worth
protecting -- an earlier velocity-dependent gain layered a persistent offset
over this map, and the resulting history dependence (the same hand pose landing
up to 178 px apart depending on how it was approached) was far more damaging to
closed-loop control than any amount of smoothing.

So precision assistance is a *clutch*, not an acceleration curve. It is entered
explicitly, anchored to a position, bounded, and released back to the plain
absolute map. While it is off, this module emits exactly what the pre-feature
mapper emitted.
"""
import math
from collections import deque
from statistics import median
from dataclasses import dataclass, replace


def _lerp(low, high, t):
    return low + (high-low)*max(0.0, min(1.0, t))


@dataclass(frozen=True)
class PrecisionProfile:
    """Low-speed precision assistance, and the state machine that gates it.

    Speeds are in normalized frame units per second -- the units the hand point
    arrives in -- so one set of thresholds fits any camera and any desktop. A
    hand crossing the active region in a third of a second travels about
    2 units/s; deliberate work on a small target sits below .1.

    There is no high-speed gain here and there should not be. The map is
    absolute, so the whole desktop is already inside the active region: gain
    above 1 buys no reach, and the only thing it can do is displace the cursor
    from the position the map says it should be at.
    """
    gain: float = .5             # cursor travel per unit of hand travel while engaged
    enter_speed: float = .10     # engage below this robust hand speed...
    exit_speed: float = .18      # ... and let go above this one
    dwell: float = .12           # seconds below enter_speed before engaging
    window: float = .20          # seconds of history behind the robust speed
    # Cap on how far the clutch may hold the cursor from the map, in screen
    # spans. It also sets the assisted range: past limit/(1-gain) of hand
    # travel the clamp binds and the hand tracks the map one-to-one again, so
    # this is the whole of "the last few pixels" and the whole of the position
    # error the clutch can ever be responsible for.
    #
    # Small, and the rule it is sized by: the clutch may never displace the
    # cursor by more than the jitter it removes. Measured on 30 FPS traces with
    # .003-unit landmark noise, the plain map wobbles about 18 px peak-to-peak
    # and the clutch cuts that to 8, so +-15 px of displacement stays inside the
    # noise it has already taken away and cannot read as a positioning error.
    #
    # It buys no extra steadiness -- that comes from the gain and the smoothing,
    # both independent of this -- only assisted range, paid for one-for-one in
    # how far two different approaches to the same hand position can land apart.
    # At .008 that worst case is about 30 px, against the 178 px the velocity
    # gain this replaced could produce, and it covers the 30 px of hand travel a
    # real final correction onto a small target takes.
    limit: float = .008
    # Extra smoothing while engaged. This is the half of precision assistance
    # that costs nothing positionally: an EMA's steady state is its input, so
    # more of it removes landmark noise without displacing the cursor from the
    # map by so much as a pixel. Latency is the price, and it is only paid at
    # the speeds where it does not matter -- and the filter's own distance
    # scaling shortens it again as soon as the target pulls away.
    tau_scale: float = 3.0
    engage_tau: float = .05      # seconds; how tightly the offset tracks the clutch
    # Once disengaged the offset bleeds off per screen span of hand travel, not
    # per second. Removing it costs cursor motion the hand did not ask for, and
    # tying that to distance makes it a fixed small fraction of whatever
    # movement is masking it: invisible at speed, and near-frozen while the
    # hand is creeping, which is when a correction would be most obvious.
    release_span: float = .10
    pinned_tau: float = .12      # seconds; the one place a timed bleed is right


PRECISION = PrecisionProfile()


class VelocityEstimator:
    """Hand speed for deciding intent, not for scaling the pointer.

    The median of the pairwise slopes across a ~200 ms window, per axis. Least
    squares was the obvious choice and is the wrong one: it is not robust, so a
    single landmark excursion inflates the slope enough to read as a fast hand,
    and the pointer would then be kicked out of precision mode by the very
    glitch precision mode is there to absorb. A median over pairs cannot be
    moved by one bad sample, costs about fifteen divisions at this window
    length, and needs no resampling to cope with the uneven frame timing a
    webcam actually delivers.

    Deliberately lower bandwidth than the pointer. A derivative of noisy
    landmark positions has no business on the pointer's critical path; this is
    only ever compared against a threshold.
    """
    def __init__(self, profile=PRECISION):
        self.profile = profile
        self.history = deque()
        self.raw = 0.0           # instantaneous, for diagnosis only
        self.speed = 0.0         # robust, what decisions are made on

    def reset(self):
        self.history.clear()
        self.raw = 0.0
        self.speed = 0.0

    def update(self, point, now, profile=None):
        profile = profile or self.profile
        previous = self.history[-1] if self.history else None
        if previous is not None and not 0 <= now-previous[0] <= 1:
            self.reset()         # a gap or a clock step: start the window again
            previous = None
        self.history.append((now, float(point[0]), float(point[1])))
        while len(self.history) > 2 and now-self.history[0][0] > profile.window:
            self.history.popleft()
        if previous is not None and now > previous[0]:
            step = now - previous[0]
            self.raw = math.dist(point[:2], previous[1:])/step
        elif previous is None:
            self.raw = 0.0
        self.speed = self._slope()
        return self.speed

    def _slope(self):
        """Magnitude of the median pairwise velocity over the window."""
        rows = self.history
        if len(rows) < 3:
            return 0.0
        # Only pairs far enough apart in time to be informative; a pair one
        # frame apart is mostly noise divided by a small number.
        span = rows[-1][0] - rows[0][0]
        least = max(1e-3, span/3)
        slopes = [[], []]
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                gap = rows[j][0] - rows[i][0]
                if gap < least:
                    continue
                slopes[0].append((rows[j][1]-rows[i][1])/gap)
                slopes[1].append((rows[j][2]-rows[i][2])/gap)
        if not slopes[0]:
            return 0.0
        return math.hypot(median(slopes[0]), median(slopes[1]))


class PrecisionClutch:
    """Fine positioning near a target, without touching the map anywhere else.

    Engaged, the emitted target is a fixed blend of the live absolute target
    and the position the hand was at when it engaged::

        target = gain*base + (1-gain)*anchor

    which is an affine function of the *current* absolute position. Nothing is
    integrated, so landmark noise cannot accumulate into a random walk, and the
    blend attenuates that noise by `gain` into the bargain. The offset it
    implies is capped, so beyond a short local range the hand tracks the map
    one-to-one again and the clutch cannot slide the cursor somewhere the map
    does not put it.

    Disengaged, the offset bleeds to exactly zero and this is the identity: the
    caller gets the absolute target unchanged.
    """
    def __init__(self, profile=PRECISION):
        self.profile = profile
        self.velocity = VelocityEstimator(profile)
        self.active = False
        self.anchor = None       # absolute target at the moment of engaging
        self.offset = (0.0, 0.0)
        self.slow_since = None
        self.base = None
        self.when = None
        self._key = None
        self._tuned = profile

    def reset(self):
        """Drop every trace of the clutch. The next frame is the plain map."""
        self.velocity.reset()
        self.disengage()
        self.offset = (0.0, 0.0)
        self.base = None
        self.when = None

    def disengage(self):
        """Let go without discarding the offset, so it can bleed off smoothly."""
        self.active = False
        self.anchor = None
        self.slow_since = None

    @property
    def raw_speed(self):
        return self.velocity.raw

    @property
    def speed(self):
        return self.velocity.speed

    @property
    def gain(self):
        """Effective cursor-travel ratio: the clutch gain, or 1 when off."""
        return self._tuned.gain if self.active else 1.0

    @property
    def tau_scale(self):
        """Smoothing multiplier for this frame: the profile's, or 1 when off."""
        return self._tuned.tau_scale if self.active else 1.0

    def tuned(self, settings):
        """The profile with the user's precision gain folded in, cached."""
        key = getattr(settings, 'precision_gain', None)
        if key != self._key:
            self._key = key
            self._tuned = self.profile if key is None else replace(
                self.profile, gain=min(max(key, .05), 1.0))
        return self._tuned

    def track(self, point, now, settings=None):
        """Follow the hand while another state owns the pointer.

        Dragging and scrolling do not want the clutch, but they do want the
        speed history to still be there when pointing resumes, so a hand that
        was moving is not mistaken for one that has settled.
        """
        self.velocity.update(point, now, self.tuned(settings) if settings is not None
                             else self.profile)

    def gate(self, now, profile, settled=None):
        """Engage or let go, on explicit hysteresis with a dwell.

        Two thresholds and a dwell rather than a continuous curve: a threshold
        that the noise floor can straddle would flip the control mode frame to
        frame, and a gain that varies continuously with a noisy derivative puts
        that noise straight onto the cursor. Neither is worth the smoothness.
        """
        speed = self.velocity.speed
        if self.active:
            if speed > profile.exit_speed:
                self.disengage()
            return self.active
        if speed <= profile.enter_speed:
            if self.slow_since is None:
                self.slow_since = now
            elif now-self.slow_since >= profile.dwell:
                self.active = True
                # Anchored where the cursor actually is, so the origin of fine
                # movement is a filtered position rather than one frame's
                # landmark error, and shifted by whatever offset is still
                # bleeding off, so engaging is exactly continuous: the clutch
                # asks for the offset the pointer already has.
                origin = tuple(settled) if settled is not None else self.base
                lead = 1 - profile.gain
                self.anchor = tuple(o + (f/lead if lead > 1e-6 else 0.0)
                                    for o, f in zip(origin, self.offset))
                self.slow_since = None
        else:
            self.slow_since = None
        return self.active

    def apply(self, point, base, now, dt, span, pinned=(False, False),
              settled=None, settings=None):
        """The target to point at: the absolute one, or a clutched one.

        `settled` is where the cursor actually is -- the filtered position -- and
        is what the clutch anchors to. Anchoring to the raw mapped point instead
        would take a single noisy landmark sample as the origin of all subsequent
        fine movement, which puts that one sample's error (up to a third of the
        offset budget) into every position the clutch then produces.

        `pinned` says, per axis, that the base map has saturated at the edge of
        the active region. The offset is driven to zero there, because pushing
        the hand further into the margin is a request to keep going and an
        offset frozen against a saturated axis would put the last strip of the
        desktop out of reach.
        """
        profile = self.tuned(settings) if settings is not None else self.profile
        self.velocity.update(point, now, profile)
        previous, self.base = self.base, tuple(base)
        gap = self.when is None or now-self.when > max(2.5*dt, .05)
        self.when = now
        if gap:
            # Something else had the pointer -- a drag, a throw, a stroke. Do
            # not read the jump in the base as hand movement, and do not carry a
            # clutch across the handover.
            self.disengage()
            self.offset = (0.0, 0.0)
            return tuple(base)
        engaged = self.gate(now, profile, settled)
        limit = profile.limit*span
        if engaged:
            wanted = [max(-limit, min(limit, -(1-profile.gain)*(b-a)))
                      for b, a in zip(base, self.anchor)]
            rate = 1 - math.exp(-max(.001, dt)/max(.001, profile.engage_tau))
        else:
            wanted = [0.0, 0.0]
            travel = math.dist(base, previous) if previous else 0.0
            rate = 1 - math.exp(-travel/max(1e-6, profile.release_span*span))
        # Driven to zero against a saturated axis, whether engaged or not, and
        # on a clock: the hand is pinned, so there is no travel to bleed against.
        timed = 1 - math.exp(-max(.001, dt)/max(.001, profile.pinned_tau))
        offset = []
        for axis, (o, w) in enumerate(zip(self.offset, wanted)):
            if pinned[axis]:
                w, step = 0.0, timed
            else:
                step = rate
            offset.append(o + (w-o)*step)
        # Snapped to zero once it is under a quarter pixel, so "disengaged"
        # really does mean the emitted target is the absolute target, exactly.
        self.offset = tuple(0.0 if not engaged and abs(o) < .25 else o for o in offset)
        return (base[0]+self.offset[0], base[1]+self.offset[1])


class AdaptiveEMA:
    def __init__(self):
        self.value = None

    def reset(self, point=None):
        self.value = point

    def update(self, point, dt, tau, deadzone):
        if self.value is None:
            self.value = point
        distance = math.dist(point, self.value)
        if distance > deadzone:
            alpha = 1 - math.exp(-max(.001, dt) / (tau / (1 + distance / 70)))
            self.value = tuple(a + alpha * (b-a) for a, b in zip(self.value, point))
        return self.value


@dataclass(frozen=True)
class DragProfile:
    """Alpha-beta tracking for a drag anchor.

    Alpha is chosen per frame from how big the residual is, not fixed. A fixed
    value cannot serve both ends of this job, and both failures were measured:

    * At .72 -- the value first shipped -- a drag held still shimmered twice as
      much as the pre-feature filter did (10 px mean step against 4.6), because
      an alpha that high simply passes landmark noise through.
    * At a fixed .30 the still hand was quiet, but a 4500 px/s drag lagged 85 px
      for six frames and the lag then grew past the outlier limit. That is a
      trap rather than a wobble: a rejected frame froze the tracked velocity, a
      frozen velocity froze the limit, and the drag never recovered -- it lost
      the ability to throw at all, because every sample was marked unclean.

    So a residual at the noise floor gets `alpha_still`, one that plainly means
    the hand is moving gets `alpha_moving`, and they are blended on
    d^2/(d^2+knee^2) so the two separate sharply rather than trading off. That
    is the same instinct as the pointer's AdaptiveEMA shortening its own time
    constant with distance, and it is why a low alpha costs nothing here: the
    velocity feed-forward, not alpha, is what tracks steady motion.

    Beta follows from whatever alpha came out at, as alpha^2/(2-alpha), which
    measures as the settling-time optimum -- 667 ms at alpha .30 against 833 ms
    at beta .04 and 1400 ms at beta .02. The .08 originally paired with alpha
    .20 sat 3.6x above that relation and paid for it, overshooting a 60 px
    correction by 35% instead of 19%, and doing it hardest exactly when
    confidence was lowest and the estimate was supposed to be settling onto a
    trajectory rather than arguing with the measurement.

    Confidence scales how far toward `alpha_moving` a given residual may reach,
    so a degraded pose leans on its trajectory instead of chasing landmarks.
    """
    alpha_still: float = .12     # correction for a residual that is only noise
    alpha_moving: float = .80    # ... and for one that is plainly hand movement
    knee: float = .022           # the crossover between them, in screen spans
    # Generous on purpose. A hand really can cross the desktop in a sixth of a
    # second, and a limit tight enough to second-guess that fights real motion
    # instead of glitches. It is measured against how far the *measurement* has
    # been moving, never against the filter's own tracked speed: a limit that
    # depends on the filter can be starved by its own rejections, which is
    # exactly the lockout described above.
    outlier_span: float = 4.0    # implausible residual speed, in screen spans/s
    outlier_scale: float = 2.5   # ... plus this multiple of the measured speed
    step_tau: float = .10        # seconds; smoothing on that measured speed
    # And the limit widens with each consecutive rejection, which is what makes
    # a lockout structurally impossible: a glitch by definition does not
    # persist, so anything that keeps arriving is real motion and is admitted
    # within a frame or two however fast it is.
    outlier_relief: float = 1.0  # extra limit per consecutive rejection
    outlier_confidence: float = .15   # a clamped frame is treated as degraded
    predict_tau: float = .12     # seconds; blind extrapolation decays this fast
    quality_tau: float = .10     # seconds; time constant of the quality readout


DRAG = DragProfile()


class DragEstimator:
    """Constant-velocity estimate of where the dragged anchor is.

    Deliberately not a heavier low-pass: latency is what makes a dragged window
    feel detached from the hand, so stability comes from rejecting implausible
    single-frame excursions and from trusting the recent trajectory when the
    landmarks stop being trustworthy, rather than from smoothing everything.
    """
    def __init__(self, profile=DRAG):
        self.profile = profile
        self.position = None
        self.velocity = (0.0, 0.0)
        self.measured = None     # last measurement offered, accepted or not
        self.quality = 0.0       # smoothed confidence of the measurements used
        self.rejected = False    # was the last measurement clamped as an outlier
        self.blind = 0.0         # seconds since the last accepted measurement
        self.step = 0.0          # smoothed speed of the measurement itself, px/s
        self.rejections = 0      # consecutive clamped frames; widens the limit

    def reset(self, point=None):
        self.position = tuple(point) if point is not None else None
        self.velocity = (0.0, 0.0)
        self.measured = tuple(point) if point is not None else None
        self.quality = 1.0 if point is not None else 0.0
        self.rejected = False
        self.blind = 0.0
        self.step = 0.0
        self.rejections = 0

    def _decay_quality(self, dt, toward):
        alpha = 1 - math.exp(-max(.001, dt)/max(.001, self.profile.quality_tau))
        self.quality += alpha*(toward-self.quality)

    def predict(self, dt, blind=False):
        """Advance the estimate by dt. `blind` means no measurement is coming."""
        if self.position is None:
            return None
        dt = max(0.0, dt)
        if blind:
            # Damped, so a latched drag drifts on for a frame or two rather than
            # flying off on a stale velocity if the hand is really gone.
            keep = math.exp(-dt/max(.001, self.profile.predict_tau))
            self.velocity = tuple(v*keep for v in self.velocity)
            self.blind += dt
            self._decay_quality(dt, 0.0)
        self.position = tuple(p + v*dt for p, v in zip(self.position, self.velocity))
        return self.position

    def outlier_limit(self, dt, span):
        """How far the anchor could plausibly have moved in dt."""
        relief = 1 + self.profile.outlier_relief*self.rejections
        return (self.profile.outlier_span*span
                + self.profile.outlier_scale*self.step)*max(1e-3, dt)*relief

    def weights(self, distance, confidence, span):
        """(alpha, beta) for a residual of this size at this confidence."""
        knee = max(1e-6, self.profile.knee*span)
        reach = distance*distance/(distance*distance + knee*knee)
        alpha = _lerp(self.profile.alpha_still, self.profile.alpha_moving,
                      reach*max(0.0, min(1.0, confidence)))
        return alpha, alpha*alpha/(2-alpha)

    def correct(self, measured, dt, confidence, span):
        """Fold in a measurement. Returns (position, accepted)."""
        previous, self.measured = self.measured, tuple(measured)
        if self.position is None:
            self.reset(measured)
            return self.position, True
        dt = max(1e-3, dt)
        confidence = max(0.0, min(1.0, confidence))
        residual = tuple(m-p for m, p in zip(measured, self.position))
        distance = math.hypot(*residual)
        # Judged against the speed established *before* this frame. Letting the
        # current measurement into that figure first would let an excursion
        # widen the very limit it is being tested against -- the same shape of
        # mistake as a limit that can be starved, in the opposite direction.
        limit = self.outlier_limit(dt, span)
        accepted = distance <= limit
        if accepted and previous is not None:
            fresh = math.dist(measured, previous)/dt
            rate = 1 - math.exp(-dt/max(.001, self.profile.step_tau))
            # Rising at once and falling smoothly: the limit has to be wide
            # enough for a movement on the frame it starts, not one frame later.
            self.step = max(fresh, self.step + rate*(fresh-self.step))
        if not accepted and distance > 0:
            # Clamped rather than discarded: a genuinely fast drag still tracks,
            # at a rate a hand could reach, while a landmark swap cannot
            # teleport the dragged object.
            residual = tuple(r*limit/distance for r in residual)
            confidence = min(confidence, self.profile.outlier_confidence)
        alpha, beta = self.weights(math.hypot(*residual), confidence, span)
        # A residual that follows a blind stretch, or one large enough to have
        # been clamped, is a position correction and nothing else. Feeding it to
        # the velocity path would read a hand that was merely re-found as a hand
        # travelling at the speed of the gap -- which is both the overshoot that
        # sends a reacquired drag past its target and, downstream, exactly the
        # fake fling velocity a reacquisition must never be able to invent.
        corrective = self.blind > 0 or not accepted
        self.position = tuple(p + alpha*r for p, r in zip(self.position, residual))
        if not corrective:
            self.velocity = tuple(v + beta*r/dt for v, r in zip(self.velocity, residual))
        self.rejected = not accepted
        self.rejections = 0 if accepted else self.rejections+1
        self.blind = 0.0
        self._decay_quality(dt, confidence if accepted else 0.0)
        return self.position, accepted


class CursorMapper:
    def __init__(self, bounds, screens=None):
        self.bounds = bounds
        self.screens = tuple(screens) if screens else (bounds,)
        self.filter = AdaptiveEMA()
        self.precision = PrecisionClutch()
        self.drag = DragEstimator()

    @property
    def span(self):
        """Widest desktop dimension; the scale the clutch and outlier limits use."""
        return max(self.bounds[2:])

    def reset(self, point=None):
        """Drop every pointer estimate. Used on pause and on tracking loss."""
        self.filter.reset(point)
        self.precision.reset()
        self.drag.reset()

    def visible_point(self, point):
        """Project a desktop point onto the nearest actual monitor rectangle."""
        candidates = [(min(x+w-1, max(x, point[0])),
                       min(y+h-1, max(y, point[1])))
                      for x, y, w, h in self.screens]
        return min(candidates, key=lambda candidate: math.dist(point, candidate))

    def clamp(self, point):
        x, y, w, h = self.bounds
        return min(x+w-1, max(x, point[0])), min(y+h-1, max(y, point[1]))

    @staticmethod
    def fraction(v, settings):
        """Where a normalized hand coordinate lands across the active region.

        Clamped to 0..1, so a value at either end also says the map has
        saturated and the hand is pushing into the margin.
        """
        return min(1, max(0, (((v-settings.margin)/(1-2*settings.margin))-.5)*settings.sensitivity+.5))

    def target(self, point, settings):
        x, y, width, height = self.bounds
        return (x + self.fraction(point[0], settings)*(width-1),
                y + self.fraction(point[1], settings)*(height-1))

    def update(self, point, settings, dt, now=None):
        """Pointing: the absolute map, optional precision clutch, then smoothing.

        With the clutch off this is the pre-feature path exactly, down to the
        filter arguments. `now` is what makes a hand speed measurable, so
        callers with no clock of their own (calibration, direct mapping checks)
        get the plain map.
        """
        target = self.target(point, settings)
        deadzone, smoothing = settings.deadzone, settings.smoothing
        if now is not None and getattr(settings, 'precision_assist', False):
            pinned = tuple(self.fraction(v, settings) in (0, 1) for v in point[:2])
            target = self.precision.apply(point, target, now, dt, self.span, pinned,
                                          self.filter.value, settings)
            # The clutch blends the live target with a fixed anchor, so it
            # attenuates landmark noise by exactly its gain. Scaling the
            # deadzone by the same factor keeps the noise-to-deadband ratio the
            # legacy path had, while making the step the cursor moves in that
            # much finer -- which is the part a small target actually needs.
            deadzone *= self.precision.gain
            smoothing *= self.precision.tau_scale
        return self.clamp(self.filter.update(target, dt, smoothing, deadzone))
