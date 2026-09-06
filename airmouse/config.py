from dataclasses import dataclass, asdict
import json
from pathlib import Path

CONFIG_PATH = Path.home() / '.config' / 'airmouse' / 'settings.json'

@dataclass
class Settings:
    camera: int = 0
    margin: float = .18
    sensitivity: float = 1.0
    smoothing: float = .09
    deadzone: float = 2.5
    drag_deadzone: float = 10.0
    pinch: float = .28
    release: float = .42
    debounce: float = .08
    cooldown: float = .35
    dwell: float = .35
    left: bool = True
    right: bool = True
    scroll: bool = True

    def validate(self):
        limits = {'camera': (0, 32), 'margin': (.05, .4), 'sensitivity': (1, 2),
                  'smoothing': (.01, .4), 'deadzone': (0, 20), 'drag_deadzone': (2, 40), 'pinch': (.1, .5),
                  'release': (.15, .8), 'debounce': (.03, .5), 'cooldown': (.1, 2), 'dwell': (.2, 2)}
        for key, (lo, hi) in limits.items():
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not lo <= value <= hi:
                raise ValueError(f'Invalid {key}')
        if self.release <= self.pinch:
            raise ValueError('Release threshold must exceed pinch threshold')
        for key in ('left', 'right', 'scroll'):
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
