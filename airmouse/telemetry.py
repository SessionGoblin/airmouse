"""What the control loop decided this frame, for the tuning diagnostics pane.

Velocity-dependent gain and a latched drag are both invisible from outside:
they change how far the pointer moved and whether a button stayed down, and
neither shows up in the state name. This is the readout that makes them
inspectable while tuning.

Nothing here prints. The controller assigns a handful of fields per frame and
the diagnostics pane formats them only while it is open, so a production run
pays those assignments and writes no logs.
"""
from dataclasses import dataclass, field


@dataclass
class Telemetry:
    state: str = ''
    # Hand speed in normalized frame units per second: the instantaneous
    # frame-to-frame figure, and the robust window estimate that mode decisions
    # are actually made on. A large gap between them is the noise floor.
    raw_speed: float = 0.0
    speed: float = 0.0
    gain: float = 1.0              # effective cursor-travel ratio: clutch gain, or 1
    precision: bool = False        # is the precision clutch engaged
    anchor: tuple = None           # where it engaged, or None
    offset: tuple = (0.0, 0.0)     # what it is displacing the cursor by
    absolute: tuple = None         # the plain absolute map's target
    pointer: tuple = None          # what was actually emitted
    # |pointer - absolute|. Outside precision mode this must be zero: anything
    # else means the map has stopped being memoryless, which is the failure the
    # clutch design exists to make impossible.
    deviation: float = 0.0
    landmark: tuple = None         # the raw pointing landmark, normalized
    # Click acquisition: the pinch is what moves the pointing landmark, so this
    # is where a click that lands off target shows up.
    pinch_settled: bool = True     # has the pinch stopped closing
    drag_intent: int = 0           # consecutive frames past the drag deadzone
    drag_probe: float = 0.0        # smoothed knuckle movement since the press, px
    confidence: float = 0.0        # how firmly the drag pose is held, 0..1
    tracking: str = ''             # ok / degraded / lost
    drag_quality: float = 0.0      # smoothed confidence of measurements actually used
    measured: tuple = None         # drag anchor as measured this frame
    predicted: tuple = None        # ... and where the trajectory estimate put it
    outlier: bool = False          # the measurement was clamped as implausible
    latched: bool = False          # drag held through a frame with no usable hand
    grace: float = 0.0             # seconds it has been running blind
    release_reason: str = ''
    release_velocity: tuple = None
    throw: str = ''                # 'accepted', or why it was rejected
    counts: dict = field(default_factory=dict)   # latched frames, outliers, timeouts

    def bump(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def lines(self):
        """Formatted for the diagnostics pane, newest facts first."""
        out = [f'Hand speed {self.raw_speed:.3f} raw / {self.speed:.3f} robust '
               f'units/s • cursor ratio {self.gain:.2f}×']
        pointer = (f'Pointer: {"PRECISION" if self.precision else "absolute map"}'
                   f' • offset {self.offset[0]:+.1f},{self.offset[1]:+.1f}px'
                   f' • deviation from map {self.deviation:.1f}px')
        out.append(pointer)
        if self.state in ('pinch-down', 'dragging'):
            out.append(f'Press: pinch {"settled" if self.pinch_settled else "STILL CLOSING"}'
                       f' • knuckle moved {self.drag_probe:.1f}px'
                       f' • deadzone crossings {self.drag_intent}')
        drag = f'Drag: {self.tracking or "—"} • pinch confidence {self.confidence:.2f}'
        if self.latched:
            drag += f' • LATCHED {self.grace*1000:.0f} ms'
        if self.outlier:
            drag += ' • measurement clamped'
        out.append(drag)
        if self.measured or self.predicted:
            def show(point):
                return f'({point[0]:.0f},{point[1]:.0f})' if point else '—'
            out.append(f'Anchor measured {show(self.measured)} / estimated '
                       f'{show(self.predicted)} • quality {self.drag_quality:.2f}')
        if self.release_reason:
            velocity = (f'{self.release_velocity[0]:.0f},{self.release_velocity[1]:.0f} px/s'
                        if self.release_velocity else 'none')
            out.append(f'Last release: {self.release_reason} • velocity {velocity}'
                       + (f' • throw {self.throw}' if self.throw else ''))
        if self.counts:
            out.append('Since resume: ' + ', '.join(f'{k} {v}' for k, v in sorted(self.counts.items())))
        return out
