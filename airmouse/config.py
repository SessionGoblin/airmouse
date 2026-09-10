from dataclasses import dataclass, asdict, fields
import json
from pathlib import Path

CONFIG_PATH = Path.home() / '.config' / 'airmouse' / 'settings.json'

@dataclass
class Settings:
    camera: int = 0
    camera_width: int = 640
    camera_height: int = 480
    margin: float = .18
    sensitivity: float = 1.0
    smoothing: float = .09
    deadzone: float = 2.5
    drag_deadzone: float = 10.0
    pinch: float = .32
    release: float = .42
    debounce: float = .05
    cooldown: float = .25
    dwell: float = .35
    left: bool = True
    right: bool = True
    scroll: bool = True
    hold_fps: bool = True
    custom: bool = True
    two_hands: bool = True
    strokes: bool = True
    fling: bool = True
    pointer_side: str = 'right'
    # Longer than the pinch debounce: a custom pose can run a command, so a
    # false positive costs more than a stray click.
    custom_dwell: float = .45
    # Low-speed precision assistance. Off the clutch, the pointer is the plain
    # absolute map -- `sensitivity` means exactly what it always meant, and
    # there is no speed at which the map stops being memoryless. Engaged, the
    # cursor moves `precision_gain` per unit of hand travel over a short local
    # range. The thresholds and dwell live in mapping.PrecisionProfile.
    precision_assist: bool = True
    precision_gain: float = .45
    # How long an established drag survives frames it cannot read at all. Short
    # enough that a hand actually leaving does not hold the button for a
    # noticeable beat, long enough to swallow the two or three frames a detector
    # drops when a hand turns or occludes its own thumb.
    drag_grace: float = .15

    def validate(self):
        for key in ('camera_width', 'camera_height'):
            value = getattr(self, key)
            if type(value) is not int or not 160 <= value <= 7680:
                raise ValueError(f'Invalid {key}')
        limits = {'camera': (0, 32), 'margin': (.05, .4), 'sensitivity': (1, 2),
                  'smoothing': (.01, .4), 'deadzone': (0, 20), 'drag_deadzone': (2, 40), 'pinch': (.1, .5),
                  'release': (.15, .8), 'debounce': (.03, .5), 'cooldown': (.1, 2), 'dwell': (.2, 2),
                  'custom_dwell': (.15, 3), 'precision_gain': (.1, 1),
                  'drag_grace': (0, .5)}
        for key, (lo, hi) in limits.items():
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not lo <= value <= hi:
                raise ValueError(f'Invalid {key}')
        if self.release <= self.pinch:
            raise ValueError('Release threshold must exceed pinch threshold')
        if self.pointer_side not in ('left', 'right', 'auto'):
            raise ValueError('Invalid pointer_side')
        for key in ('left', 'right', 'scroll', 'hold_fps', 'custom', 'two_hands', 'strokes',
                    'fling', 'precision_assist'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'Invalid {key}')
        self.camera = int(self.camera)
        return self

    @classmethod
    def load(cls):
        try:
            stored = json.loads(CONFIG_PATH.read_text())
        except (OSError, ValueError):
            return cls()
        if not isinstance(stored, dict):
            return cls()
        # Keys this build does not know are dropped rather than rejected, and
        # keys it knows that the file lacks keep their default. A settings file
        # written by a newer build therefore still loads everything it shares
        # with this one, instead of the whole file being discarded over one
        # field -- which is how the older behaviour lost a user's calibration.
        known = {f.name for f in fields(cls)}
        # The velocity-gain settings this replaced: an offset layered over the
        # absolute map made the pointer history-dependent, so the curve is gone
        # and only the low-speed end survives, as an explicit clutch. Carry the
        # switch and the slow-end gain across; `gain_max` described high-speed
        # lead, which no longer exists to configure.
        for stale, current in (('pointer_gain', 'precision_assist'),
                               ('gain_min', 'precision_gain')):
            if stale in stored and current not in stored:
                stored[current] = stored[stale]
        try:
            return cls(**{k: v for k, v in stored.items() if k in known}).validate()
        except (ValueError, TypeError):
            return cls()

    def save(self):
        self.validate()
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CONFIG_PATH.with_suffix('.tmp')
        temporary.write_text(json.dumps(asdict(self), indent=2))
        temporary.replace(CONFIG_PATH)
