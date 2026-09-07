from dataclasses import dataclass, asdict
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
    pointer_side: str = 'right'
    # Longer than the pinch debounce: a custom pose can run a command, so a
    # false positive costs more than a stray click.
    custom_dwell: float = .45

    def validate(self):
        for key in ('camera_width', 'camera_height'):
            value = getattr(self, key)
            if type(value) is not int or not 160 <= value <= 7680:
                raise ValueError(f'Invalid {key}')
        limits = {'camera': (0, 32), 'margin': (.05, .4), 'sensitivity': (1, 2),
                  'smoothing': (.01, .4), 'deadzone': (0, 20), 'drag_deadzone': (2, 40), 'pinch': (.1, .5),
                  'release': (.15, .8), 'debounce': (.03, .5), 'cooldown': (.1, 2), 'dwell': (.2, 2),
                  'custom_dwell': (.15, 3)}
        for key, (lo, hi) in limits.items():
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not lo <= value <= hi:
                raise ValueError(f'Invalid {key}')
        if self.release <= self.pinch:
            raise ValueError('Release threshold must exceed pinch threshold')
        if self.pointer_side not in ('left', 'right'):
            raise ValueError('Invalid pointer_side')
        for key in ('left', 'right', 'scroll', 'hold_fps', 'custom', 'two_hands'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'Invalid {key}')
        self.camera = int(self.camera)
        return self

    @classmethod
    def load(cls):
        try:
            return cls(**json.loads(CONFIG_PATH.read_text())).validate()
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self):
        self.validate()
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CONFIG_PATH.with_suffix('.tmp')
        temporary.write_text(json.dumps(asdict(self), indent=2))
        temporary.replace(CONFIG_PATH)
