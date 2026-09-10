import threading
import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from .gestures import GestureMachine, State
from .mapping import CursorMapper
from .telemetry import Telemetry
from . import gestures as gestures_module, poses as poses_module, strokes as strokes_module

# Fling. Releasing a pinch mid-motion keeps the button held and coasts the
# pointer: a compositor stops moving a window the moment the button comes up,
# so animating a throw means holding through the coast and releasing at the end.
TRAIL = .20           # seconds of drag positions kept for the velocity estimate
SETTLE = .06          # ignored before release; opening a pinch drags the fingertip
MIN_SPAN = .04        # minimum time between the two samples used
FLING_MIN = 600       # px/s below which a release is just a release
FLING_STOP = 80       # px/s at which the coast ends
FLING_TAU = .30       # seconds; velocity decays as exp(-dt/tau)
FLING_LIMIT = 1.2     # coast distance cap, as a fraction of the widest screen span
FLING_DEADLINE = .8   # hard stop, so the button can never be held indefinitely
# How the deadzone decides that a press is really a drag. On a 1920 desktop the
# mapping is about 3000 px per unit of hand travel, so even .002 of landmark
# noise is 6 px against a 10 px deadzone -- and with a fresh comparison every
# frame, noise alone crossed it on 13 clicks in 30. Averaging a few frames and
# then asking for two crossings in a row puts that essentially at nil, while a
# hand that is actually moving still crosses within about two frames.
DRAG_PROBE_TAU = .09
DRAG_PROBE_FRAMES = 4
# ... except when the hand has plainly gone somewhere. A displacement this many
# times the deadzone is not something noise produces, so it starts the drag on
# the frame it arrives and a decisive drag keeps close to its old acquisition
# speed -- 49 ms against the 33 ms of a raw comparison, where a gentle one pays
# about 110 ms for not being confusable with a click.
DRAG_PROBE_CLEAR = 2.5
# Positions emitted while tracking was unusable, and for this long after it came
# back, take no part in a release velocity. A predicted anchor and the first
# correction back onto a reacquired hand are both perfectly smooth motion that
# the user never made, and a throw built from either is a throw nobody threw.
REACQUIRE_SETTLE = .08


@dataclass
class Throw:
    point: tuple
    velocity: tuple
    limit: float
    deadline: float
    travel: float = 0.0

class Controller:
    """Serialized control gate, including emergency release from keyboard thread."""
    def __init__(self, settings, bounds, backend=None, screens=None,
                 library=None, dispatcher=None, stroke_library=None):
        self.settings, self.backend = settings, backend
        self.mapper = CursorMapper(bounds, screens)
        self.machine = GestureMachine(library)
        self.dispatcher = dispatcher
        self.stroke_library = stroke_library
        self.capture = strokes_module.StrokeCapture()
        # Frame aspect, so a drawn path is compared in the same stretched space
        # the poses use. Set by the worker once the frame size is known.
        self.aspect = 4/3
        self.last_stroke = (None, 0.0)
        # Name of the modifier pose currently held. Pointer poses and strokes
        # scope to it, which is what makes the off hand a modifier rather than
        # a second pose slot.
        self.mode = None
        self.mode_latch = poses_module.PoseLatch()
        # The mode that was held while a stroke was being drawn. Recognition
        # happens on the frame the gate opens, by which point the gate pose is
        # already released and self.mode has cleared.
        self.stroke_mode = None
        self.gate_open = False
        self.throw = None
        self.trail = deque()
        self.lock = threading.RLock()
        self.enabled = False
        self.last = None
        self.target = None
        self.last_seen = None
        self.drag_origin = None
        self.drag_cursor = None
        self.drag_started = False
        # Low-passed mapped fingertip, used only to decide whether the hand has
        # really moved far enough to be dragging.
        self.drag_probe = None
        self.drag_anchor = None
        self.drag_intent = 0
        self.raw_point = None
        # Trail samples timestamped before this are not clean; see REACQUIRE_SETTLE.
        self.trail_dirty_until = 0.0
        self.absolute = None
        self.telemetry = Telemetry()

    def pause(self):
        with self.lock:
            self.enabled = False
            try:
                if self.backend: self.backend.up()
            finally:
                self.machine.reset()
                self.mapper.reset()
                self.last = None
                self.target = None
                self.drag_origin = None
                self.drag_cursor = None
                self.drag_started = False
                self.drag_probe = None
                self.drag_anchor = None
                self.drag_intent = 0
                self.raw_point = None
                self.capture.reset()
                self.mode = None
                self.mode_latch.reset()
                self.stroke_mode = None
                self.gate_open = False
                # backend.up() above already released; drop the coast so an
                # emergency stop cannot leave the button held.
                self.throw = None
                self.trail.clear()
                self.trail_dirty_until = 0.0
                self.absolute = None
                self.telemetry = Telemetry()

    def resume(self):
        with self.lock:
            if self.backend is None: return False
            self.pause()
            self.enabled = True
            return True

    def process(self, features, now, modifier=None):
        with self.lock:
            if features:
                valid = all(math.isfinite(v) for v in (*features.point, features.left, features.right))
                # A quarter-frame discontinuity in a single short interval is
                # unsafe to interpret as a click or a change of hand identity.
                jump = (valid and self.raw_point is not None and self.last is not None
                        and now-self.last < .12 and math.dist(features.point, self.raw_point) > .25)
                self.raw_point = features.point if valid else None
                if not valid or jump:
                    features = None
            else:
                self.raw_point = None
            unusable = features is None or (self.last is not None and now-self.last > .3)
            # An established drag rides out a frame or two it cannot read; only
            # the machine decides, so its view of the latch and this one agree.
            latched = unusable and self.enabled and self.machine.drag_hold(now, self.settings)
            if unusable and not latched:
                # A hand lost mid-coast ends it here rather than coasting on
                # blind: this is a tracking failure, not an intent. Releasing
                # directly, rather than through the action loop, is what keeps a
                # lost hand from ever reaching the throw path.
                self.end_throw()
                held = self.machine.down
                if self.backend and held:
                    self.telemetry.bump('tracking timeouts')
                    self.backend.up()
                self.machine.reset(paused=not self.enabled)
                if held:
                    # After the reset, which clears it: a drag that ended here
                    # ended because the hand stopped being readable.
                    self.machine.release_reason = (
                        gestures_module.RELEASE_TIMEOUT if self.enabled
                        else gestures_module.RELEASE_PAUSED)
                self.mapper.reset()
                self.drag_origin = None
                self.drag_cursor = None
                self.drag_started = False
                self.drag_probe = None
                self.drag_anchor = None
                self.drag_intent = 0
                self.trail.clear()
            dt = min(.1, now-self.last) if self.last is not None else 1/30
            self.last = now
            self.last_seen = now if features else None
            self.update_mode(modifier, now)
            # The modifier hand gates stroke drawing. Without a gate a
            # recognizer running over the cursor path would fire during
            # ordinary pointing, because here the pointer is the hand.
            gating = bool(self.settings.strokes and self.enabled
                          and self.drawing_gate(modifier))
            if gating:
                self.stroke_mode = self.mode
            drawn = self.capture.update(gating, features.point if features else None)
            if drawn is not None:
                self.recognize(drawn, self.stroke_mode)
                self.stroke_mode = None
            if gating:
                # Freeze the cursor while drawing, and drop a drag rather than
                # smearing the window along the stroke.
                self.end_throw()
                for action in self.machine.release():
                    if self.backend: self.backend.up()
                self.machine.state = State.DRAWING
                self.observe(now)
                return State.DRAWING.value
            if latched:
                return self.hold_drag(now, dt)
            if self.throw is not None:
                # Re-pinching catches the throw in flight; otherwise the coast
                # owns the pointer and the machine is not stepped at all.
                if features and self.settings.left and features.left < self.settings.pinch:
                    self.end_throw()
                elif self.advance_throw(dt):
                    self.observe(now)
                    return State.THROWN.value
            if self.mapper.filter.value is None and features:
                # Slew from current desktop position on X11; virtual absolute pointer
                # starts at centre and approaches the hand slowly on acquisition.
                origin = (self.mapper.bounds[0]+self.mapper.bounds[2]/2,
                          self.mapper.bounds[1]+self.mapper.bounds[3]/2)
                if self.target is not None:
                    origin = self.target
                if self.backend and hasattr(self.backend, 'mouse'):
                    origin = tuple(self.backend.mouse.position)
                self.mapper.filter.reset(self.mapper.visible_point(origin))
            actions = self.machine.step(features, now, self.settings, self.enabled, self.mode)
            for action in actions:
                # Ahead of the backend guard: a keystroke or shell binding does
                # not need a pointer device to be useful.
                if action[0] == 'custom':
                    self.custom(action[1])
                    continue
                if not self.backend: continue
                if action[0] == 'move':
                    previous = self.mapper.filter.value
                    point = action[1]
                    clean = True
                    if self.machine.down and self.drag_origin is not None:
                        raw = self.mapper.target(point, self.settings)
                        if not self.drag_started:
                            # Whether this press is a drag is decided on the
                            # knuckle, averaged over a couple of frames: the
                            # fingertip is still swinging into the pinch, and its
                            # own noise is the size of the deadzone.
                            here = self.hand_point(features, raw)
                            probe = 1 - math.exp(-max(.001, dt)/DRAG_PROBE_TAU)
                            self.drag_probe = tuple(
                                a+(b-a)*probe for a, b in zip(self.drag_probe, here))
                            if not self.machine.pinch_settled:
                                # Still closing. Hold the cursor where the click
                                # landed, and let both anchors follow the hand,
                                # so a drag is measured from where the pinch
                                # ended up rather than from where it started.
                                self.drag_origin = raw
                                self.drag_anchor = self.drag_probe
                                self.machine.state = State.PINCH_DOWN
                                continue
                            moved = tuple(b-a for a, b in zip(self.drag_anchor,
                                                              self.drag_probe))
                            if math.hypot(*moved) < self.settings.drag_deadzone:
                                self.drag_intent = 0
                                self.machine.state = State.PINCH_DOWN
                                continue
                            self.drag_intent += 1
                            if (self.drag_intent < DRAG_PROBE_FRAMES
                                    and math.hypot(*moved)
                                    < self.settings.drag_deadzone*DRAG_PROBE_CLEAR):
                                self.machine.state = State.PINCH_DOWN
                                continue
                            # `drag_origin` stays where the pinch settled, so the
                            # movement that just broke the deadzone is part of the
                            # drag rather than being discarded with it.
                            self.drag_started = True
                        displacement = tuple(b-a for a,b in zip(self.drag_origin, raw))
                        self.machine.state = State.DRAGGING
                        # Move relative to where the pinch landed, rather than
                        # snapping to the fingertip displaced by closing it. No
                        # velocity gain here: a dragged window should stay where
                        # the hand put it, one to one.
                        target = self.mapper.clamp(tuple(a+b for a,b in zip(self.drag_cursor, displacement)))
                        desired, clean = self.drag_step(target, dt, now)
                        # No clutch here, but the hand is still moving: keep
                        # the speed history warm so pointing resumes knowing the
                        # hand was moving rather than settling.
                        self.mapper.precision.track(point, now, self.settings)
                    else:
                        # Kept alongside the emitted position: |pointer - absolute|
                        # outside precision mode is the single number that says
                        # whether the map is still memoryless, and it is worth
                        # being able to read directly rather than reconstruct.
                        self.absolute = self.mapper.target(point, self.settings)
                        desired = self.mapper.update(point, self.settings, dt, now)
                    # Limit speed during reacquisition and normal operation.
                    cap = max(self.mapper.bounds[2:]) * dt * 1.7
                    distance = math.dist(previous, desired)
                    if distance > cap:
                        desired = tuple(a+(b-a)*cap/distance for a,b in zip(previous,desired))
                    # Keep the filtered path continuous through empty desktop space,
                    # or speed limiting can trap the pointer at a monitor edge.
                    # Only emitted positions snap across gaps (including drags).
                    self.target = self.mapper.visible_point(desired)
                    self.mapper.filter.value = desired
                    self.backend.move(*(round(v) for v in self.target))
                    if self.machine.down:
                        # Emitted positions, not the raw fingertip: the throw
                        # should carry the speed the window actually had.
                        self.record_trail(now, self.target, clean)
                elif action[0] == 'down':
                    self.drag_origin = self.mapper.target(features.point, self.settings)
                    self.drag_cursor = self.target or self.mapper.filter.value
                    self.drag_started = False
                    # Seeded at the press, not at the next frame's landmark:
                    # starting it from a raw sample would put a full frame of
                    # noise into the very first deadzone comparison, which is
                    # the one that decides whether a click becomes a drag.
                    self.drag_anchor = self.hand_point(features, self.drag_origin)
                    self.drag_probe = self.drag_anchor
                    self.drag_intent = 0
                    self.trail.clear()
                    self.trail_dirty_until = 0.0
                    self.mapper.drag.reset()
                    # The drag owns the pointer from here, so hand the clutch
                    # back explicitly rather than leaving an offset parked to
                    # reappear whenever pointing resumes.
                    self.mapper.precision.reset()
                    self.backend.down()
                elif action[0] == 'up':
                    thrown = self.start_throw(now)
                    self.drag_origin = None
                    self.drag_started = False
                    self.drag_probe = None
                    self.drag_anchor = None
                    self.drag_intent = 0
                    self.mapper.drag.reset()
                    if not thrown:
                        self.backend.up()
                elif action[0] == 'scroll': self.backend.scroll(action[1])
                else: getattr(self.backend, action[0])()
            self.observe(now)
            return self.machine.state.value

    def update_mode(self, modifier, now):
        """Read the modifier hand's held pose and publish it as the mode."""
        template = None
        if (self.machine.library is not None and self.settings.custom
                and self.enabled and modifier is not None):
            template = self.machine.library.match(
                modifier.pose, modifier.orientation, modifier.chirality, role='modifier')
        fired = self.mode_latch.update(template, now, self.settings.custom_dwell)
        held = self.mode_latch.template
        self.mode = held.name if held is not None else None
        # A gate pose is a mode, not an event: firing its binding as well would
        # run the action every time drawing starts.
        if fired is not None and not fired.gates_drawing:
            self.custom(fired)

    def drawing_gate(self, modifier):
        """Whether the modifier hand is asking to draw.

        A recorded gate pose wins once one exists; the built-in pinch remains
        the fallback so drawing works before anything has been recorded. Using
        a pose as the gate also means the mode during a stroke is the gating
        pose, which is what lets strokes be scoped per gate.
        """
        if modifier is None:
            self.gate_open = False
            return False
        gates = self.machine.library.gates() if self.machine.library is not None else set()
        if gates:
            self.gate_open = self.mode in gates
        else:
            # Hysteresis, the same shape the click gesture uses. A bare
            # threshold lets pinch noise flicker the gate and chop one stroke
            # into several too-short ones.
            limit = self.settings.release if self.gate_open else self.settings.pinch
            self.gate_open = modifier.left < limit
        return self.gate_open

    def hand_point(self, features, fallback):
        """Where the hand is, in desktop pixels, ignoring what it is pointing at.

        The knuckle through the same mapping as the pointer, so `drag_deadzone`
        keeps meaning "this many screen pixels of hand movement". Falls back to
        the pointing landmark when a caller supplies bare features without it.
        """
        palm = getattr(features, 'palm', None) if features else None
        return self.mapper.target(palm, self.settings) if palm else fallback

    def drag_step(self, measured, dt, now):
        """Where the dragged anchor is, given this frame's measurement.

        Returns (position, clean), where clean says whether the measurement was
        plausible enough for the position to count toward a release velocity.

        The estimate is corrected toward the measurement in proportion to how
        firmly the pinch is being held, so a pose going uncertain slides onto
        its own recent trajectory instead of chasing landmarks that are
        drifting, and slides back off it as the pose firms up -- converging
        rather than snapping. Stability here is bought by rejecting implausible
        excursions, not by smoothing hard enough to survive them, because
        smoothing is exactly what makes a dragged window feel detached.
        """
        estimator = self.mapper.drag
        if estimator.position is None:
            # First frame past the deadzone: the drag starts where it broke, so
            # seed there rather than correcting toward it from the press point.
            estimator.reset(measured)
            return estimator.position, now >= self.trail_dirty_until
        estimator.predict(dt)
        position, accepted = estimator.correct(
            measured, dt, self.machine.confidence, self.mapper.span)
        if not accepted:
            self.telemetry.bump('clamped measurements')
        return position, accepted and now >= self.trail_dirty_until

    def hold_drag(self, now, dt):
        """One frame of a drag latched through tracking we cannot use.

        The anchor coasts on its damped trajectory, so a dropped frame or two
        reads as the window continuing to move rather than as a stall. Nothing
        here can release the button, and every position it emits is marked
        unclean: a throw must never be assembled out of motion the user did not
        make.
        """
        state = self.machine.latch(now, self.drag_started)
        self.telemetry.bump('latched frames')
        self.trail_dirty_until = now + REACQUIRE_SETTLE
        estimator = self.mapper.drag
        if estimator.position is None:
            estimator.blind += dt       # nothing to coast: the drag never moved
        else:
            point = self.mapper.clamp(estimator.predict(dt, blind=True))
            estimator.position = point
            if self.drag_started and self.backend:
                self.target = self.mapper.visible_point(point)
                self.mapper.filter.value = point
                self.backend.move(*(round(v) for v in self.target))
                self.record_trail(now, self.target, clean=False)
        self.observe(now, latched=True)
        return state.value

    def record_trail(self, now, point, clean=True):
        """Keep one emitted drag position for the release-velocity estimate."""
        self.trail.append((now, tuple(point), bool(clean)))
        while self.trail and now-self.trail[0][0] > TRAIL:
            self.trail.popleft()

    def observe(self, now, latched=False):
        """Publish this frame's control decisions for the diagnostics pane."""
        telemetry, precision, drag = self.telemetry, self.mapper.precision, self.mapper.drag
        telemetry.state = self.machine.state.value
        telemetry.raw_speed, telemetry.speed = precision.raw_speed, precision.speed
        telemetry.gain = precision.gain
        telemetry.precision = precision.active
        telemetry.anchor = precision.anchor if precision.active else None
        telemetry.offset = precision.offset
        telemetry.absolute = self.absolute
        telemetry.pointer = self.target
        telemetry.deviation = (math.dist(self.target, self.absolute)
                               if self.target and self.absolute else 0.0)
        telemetry.confidence = self.machine.confidence
        telemetry.tracking = self.machine.tracking
        telemetry.latched = latched
        # Time since the last frame that read, which is what the grace period
        # is actually spent against.
        telemetry.grace = (max(0.0, now-self.machine.seen)
                           if self.machine.down and self.machine.seen is not None else 0.0)
        telemetry.drag_quality = drag.quality
        telemetry.outlier = drag.rejected
        telemetry.measured = drag.measured if self.machine.down else None
        telemetry.predicted = drag.position if self.machine.down else None
        telemetry.release_reason = self.machine.release_reason or ''
        telemetry.landmark = self.raw_point
        telemetry.pinch_settled = self.machine.pinch_settled
        telemetry.drag_intent = self.drag_intent
        telemetry.drag_probe = (math.dist(self.drag_probe, self.drag_anchor)
                                if self.drag_probe and self.drag_anchor else 0.0)

    def fling_velocity(self, now):
        """Pointer velocity just before a release, in px/s, or None.

        Estimated as the median of the per-frame velocities across the window
        rather than from its endpoints: one glitched landmark, or one frame the
        drag estimate had to clamp, then moves the answer by nothing instead of
        setting the throw's whole direction.

        The last SETTLE seconds are excluded -- opening a pinch pulls the index
        fingertip sideways for a frame or two, and including that curves every
        throw toward wherever the thumb went. Samples emitted while tracking
        was unusable, or just after it came back, are not clean and take no
        part at all.
        """
        usable = [(t, p) for t, p, clean in self.trail if clean and SETTLE <= now-t <= TRAIL]
        if len(usable) < 2:
            return None
        span = usable[-1][0] - usable[0][0]
        if span < MIN_SPAN:
            return None
        steps = [((b[1][0]-a[1][0])/(b[0]-a[0]), (b[1][1]-a[1][1])/(b[0]-a[0]))
                 for a, b in zip(usable, usable[1:]) if b[0]-a[0] > 1e-4]
        if len(steps) < 3:
            # Too few for a median to reject anything; the whole-span estimate
            # is the steadier answer, and is itself an average over the window.
            return ((usable[-1][1][0]-usable[0][1][0])/span,
                    (usable[-1][1][1]-usable[0][1][1])/span)
        return (median(step[0] for step in steps), median(step[1] for step in steps))

    def start_throw(self, now):
        """Begin a coast instead of releasing, if the drag earned one."""
        self.telemetry.release_velocity = None
        self.telemetry.throw = ''
        if not self.settings.fling or not self.drag_started:
            self.telemetry.throw = 'off' if not self.settings.fling else 'no drag'
            return False        # a click, or a drag that never moved, is not a throw
        if self.machine.release_reason != gestures_module.RELEASE_INTENTIONAL:
            # tracking loss != intentional release. Only an opened pinch throws;
            # a hand that went missing has no velocity it meant to impart, and
            # whatever the trail holds is not evidence of one.
            self.telemetry.throw = f'rejected: {self.machine.release_reason or "unknown release"}'
            return False
        velocity = self.fling_velocity(now)
        self.telemetry.release_velocity = velocity
        if velocity is None or math.hypot(*velocity) < FLING_MIN:
            self.telemetry.throw = 'rejected: too slow' if velocity else 'rejected: no clean trail'
            return False
        origin = self.target or self.mapper.filter.value
        if origin is None or self.backend is None:
            self.telemetry.throw = 'rejected: no pointer'
            return False
        self.throw = Throw(point=tuple(origin), velocity=velocity,
                           limit=max(self.mapper.bounds[2:])*FLING_LIMIT,
                           deadline=now+FLING_DEADLINE)
        self.trail.clear()
        self.machine.state = State.THROWN
        self.telemetry.throw = 'accepted'
        return True

    def advance_throw(self, dt):
        """Move one coast step. False once the throw is over."""
        throw = self.throw
        if throw is None:
            return False
        if (math.hypot(*throw.velocity) < FLING_STOP or throw.travel > throw.limit
                or self.last > throw.deadline):
            self.end_throw()
            return False
        step = (throw.velocity[0]*dt, throw.velocity[1]*dt)
        point = (throw.point[0]+step[0], throw.point[1]+step[1])
        x, y, w, h = self.mapper.bounds
        clamped = (min(x+w-1, max(x, point[0])), min(y+h-1, max(y, point[1])))
        throw.point = clamped
        throw.travel += math.hypot(*step)
        throw.velocity = tuple(v*math.exp(-dt/FLING_TAU) for v in throw.velocity)
        self.target = self.mapper.visible_point(clamped)
        # Keep the filter on the coast path, so the pointer reacquires from
        # where it landed rather than snapping back to the pre-throw position.
        self.mapper.filter.value = clamped
        if self.backend:
            self.backend.move(*(round(v) for v in self.target))
        if clamped != point:
            self.end_throw()        # stop at the desktop edge, not along it
            return False
        return True

    def end_throw(self):
        """Release the button the coast has been holding down."""
        if self.throw is None:
            return
        self.throw = None
        self.drag_cursor = None
        if self.backend:
            self.backend.up()

    def recognize(self, path, mode=None):
        """Score a finished stroke and run whatever it is bound to."""
        points = strokes_module.canonical(path, self.aspect)
        if self.stroke_library is None or points is None:
            self.last_stroke = (None, 0.0)
            return
        nearest, rating = self.stroke_library.nearest(points)
        self.last_stroke = (nearest.name if nearest else None, rating)
        stroke = self.stroke_library.match(points, mode=mode)
        if stroke is not None:
            self.custom(stroke)

    def custom(self, template):
        """Run a matched template's binding.

        'recenter' is handled here rather than in the dispatcher because it is
        pointer state: clearing the filter sends the cursor back through the
        same slow reacquisition path used when a hand first appears, instead of
        teleporting it.
        """
        if template.action == 'app' and template.argument == 'recenter':
            self.mapper.reset()
            self.target = None
            return
        if self.dispatcher:
            self.dispatcher.run(template)

    def close(self):
        with self.lock:
            try: self.pause()
            finally:
                if self.backend: self.backend.close()
                self.backend = None
