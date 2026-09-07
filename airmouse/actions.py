"""What a recorded pose does: a keystroke, an app command, or a shell command.

Parsing and validation are pure so a binding can be checked when it is typed
rather than failing silently the first time the pose fires. Dispatch never
raises into the vision thread: a broken binding records an error and leaves
tracking running.
"""
import os
import shlex
import subprocess
import threading

from .input import EVDEV_KEYS, PYNPUT_KEYS

KINDS = ('none', 'key', 'app', 'shell')
APP_ACTIONS = ('pause', 'toggle', 'recenter')

# Tokens both backends can express. Anything outside this is rejected at entry.
KEY_TOKENS = frozenset(EVDEV_KEYS) | frozenset(PYNPUT_KEYS)
MODIFIERS = ('ctrl', 'alt', 'shift', 'super', 'altgr')
# The app's own global hotkeys. Injecting these would be read straight back by
# the hotkey listener, so a pose could pause the app or fight it every frame.
RESERVED = ('f8', 'f12')


def parse_keys(spec):
    """'ctrl+alt+t' -> ('ctrl', 'alt', 't'). Raises ValueError on bad input."""
    tokens = [t.strip().lower() for t in str(spec).split('+') if t.strip()]
    if not tokens:
        raise ValueError('Empty shortcut')
    unknown = [t for t in tokens if t not in KEY_TOKENS]
    if unknown:
        raise ValueError(f'Unknown key: {", ".join(unknown)}')
    reserved = [t for t in tokens if t in RESERVED]
    if reserved:
        raise ValueError(f'{reserved[0].upper()} is reserved for pause / emergency stop')
    # A lone modifier is a real binding -- Super alone opens the overview on
    # GNOME and KDE -- but a combination trailing off into one is a typo.
    if len(tokens) > 1 and tokens[-1] in MODIFIERS:
        raise ValueError('Shortcut ends on a modifier; add the key it modifies')
    if len(set(tokens)) != len(tokens):
        raise ValueError('Repeated key in shortcut')
    return tuple(tokens)


def validate(kind, argument):
    """Check a binding at entry time. Returns the normalized argument."""
    if kind not in KINDS:
        raise ValueError(f'Unknown action: {kind}')
    if kind == 'none':
        return ''
    if kind == 'key':
        return '+'.join(parse_keys(argument))
    if kind == 'app':
        if argument not in APP_ACTIONS:
            raise ValueError(f'Unknown app action: {argument}')
        return argument
    if not str(argument).strip():
        raise ValueError('Empty command')
    if not shlex.split(argument):
        raise ValueError('Command parses to nothing')
    return str(argument).strip()


class Dispatcher:
    """Runs a matched template's binding.

    The keyboard device is created once, on a background thread, and only for a
    setup that actually binds a key. Building it takes real time -- evdev waits
    on udev to chmod the new node, retrying for up to two seconds -- and this
    runs on the vision thread holding the controller lock, so doing it inline
    stalled tracking long enough for the watchdog to disable control. That
    happened on the first key gesture and never again, the device being cached.
    """

    def __init__(self, app=None, keyboard_factory=None, spawn=None):
        self.app = app                      # called with 'pause' | 'toggle' | 'recenter'
        self.error = ''
        self.last = None
        self._keyboard = None
        self._factory = keyboard_factory
        self._spawn = spawn or _spawn
        self._lock = threading.Lock()
        self._priming = None

    def _build(self):
        factory = self._factory
        if factory is None:
            from .input import create_keyboard
            factory = create_keyboard
        return factory()

    def prime(self):
        """Start building the keyboard device, off whatever thread called.

        Called when a key binding exists, so the device is ready before the
        gesture that needs it rather than being built underneath it.
        """
        with self._lock:
            if self._keyboard is not None or self._priming is not None:
                return
            def build():
                try:
                    keyboard = self._build()
                except Exception as exc:
                    keyboard, self.error = None, f'keyboard unavailable: {exc}'
                with self._lock:
                    self._keyboard = keyboard
                    self._priming = None
            self._priming = threading.Thread(target=build, daemon=True)
            self._priming.start()

    def ready(self):
        with self._lock:
            return self._keyboard is not None

    def keyboard(self):
        """The device, or None while it is still being built."""
        with self._lock:
            return self._keyboard

    def run(self, template):
        """Dispatch one match. Returns True when the binding ran."""
        kind, argument = template.action, template.argument
        if not kind or kind == 'none':
            return False
        try:
            if kind == 'key':
                keyboard = self.keyboard()
                if keyboard is None:
                    # Never build it here: this is the vision thread, and the
                    # wait would stall tracking into the watchdog. Start it and
                    # let the next press land.
                    self.prime()
                    self.error = f'{template.name}: keyboard still starting up'
                    return False
                keyboard.tap(parse_keys(argument))
            elif kind == 'app':
                if argument not in APP_ACTIONS:
                    raise ValueError(f'Unknown app action: {argument}')
                if self.app:
                    self.app(argument)
            elif kind == 'shell':
                self._spawn(argument)
            else:
                raise ValueError(f'Unknown action: {kind}')
        except Exception as exc:
            # A bad binding must not take the vision thread down with it.
            self.error = f'{template.name}: {exc}'
            return False
        self.error = ''
        self.last = template.name
        return True

    def close(self):
        with self._lock:
            keyboard, self._keyboard = self._keyboard, None
        if keyboard:
            try:
                keyboard.close()
            except Exception:
                pass


def _spawn(command):
    """Launch detached, without a shell, and without inheriting our streams.

    No shell=True: the command is stored in a config file and fires on a
    gesture, so it should never grow the ability to expand globs or chain
    operators by accident.
    """
    with open(os.devnull, 'wb') as null:
        subprocess.Popen(shlex.split(command), stdin=subprocess.DEVNULL,
                         stdout=null, stderr=null, start_new_session=True)
