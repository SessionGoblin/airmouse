import threading
import math
from .gestures import GestureMachine, State
from .mapping import CursorMapper
from . import poses as poses_module, strokes as strokes_module

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
        self.lock = threading.RLock()
        self.enabled = False
        self.last = None
        self.target = None
        self.last_seen = None
        self.drag_origin = None
        self.drag_cursor = None
        self.drag_started = False
        self.raw_point = None

    def pause(self):
        with self.lock:
            self.enabled = False
            try:
                if self.backend: self.backend.up()
            finally:
                self.machine.reset()
                self.mapper.filter.reset()
                self.last = None
                self.target = None
                self.drag_origin = None
                self.drag_cursor = None
                self.drag_started = False
                self.raw_point = None
                self.capture.reset()
                self.mode = None
                self.mode_latch.reset()
                self.stroke_mode = None
                self.gate_open = False

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
            if features is None or (self.last is not None and now-self.last > .3):
                if self.backend and self.machine.down: self.backend.up()
                self.machine.reset(paused=not self.enabled)
                self.mapper.filter.reset()
                self.drag_origin = None
                self.drag_cursor = None
                self.drag_started = False
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
                for action in self.machine.release():
                    if self.backend: self.backend.up()
                self.machine.state = State.DRAWING
                return State.DRAWING.value
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
                    if self.machine.down and self.drag_origin is not None:
                        raw = self.mapper.target(point, self.settings)
                        displacement = tuple(b-a for a,b in zip(self.drag_origin, raw))
                        if not self.drag_started:
                            if math.hypot(*displacement) < self.settings.drag_deadzone:
                                self.machine.state = State.PINCH_DOWN
                                continue
                            self.drag_started = True
                        self.machine.state = State.DRAGGING
                        # Move relative to where the pinch landed, rather than
                        # snapping to the fingertip displaced by closing it.
                        target = tuple(a+b for a,b in zip(self.drag_cursor, displacement))
                        x,y,w,h = self.mapper.bounds
                        target = (min(x+w-1,max(x,target[0])), min(y+h-1,max(y,target[1])))
                        desired = self.mapper.filter.update(target,dt,self.settings.smoothing,self.settings.deadzone)
                    else:
                        desired = self.mapper.update(point, self.settings, dt)
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
                elif action[0] == 'down':
                    self.drag_origin = self.mapper.target(features.point, self.settings)
                    self.drag_cursor = self.target or self.mapper.filter.value
                    self.drag_started = False
                    self.backend.down()
                elif action[0] == 'up':
                    self.drag_origin = None
                    self.drag_started = False
                    self.backend.up()
                elif action[0] == 'scroll': self.backend.scroll(action[1])
                else: getattr(self.backend, action[0])()
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
            self.mapper.filter.reset()
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
