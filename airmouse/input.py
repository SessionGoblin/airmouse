"""Desktop adapters and global hotkeys. No gesture knowledge lives here."""
import os
import select
import threading
from glob import glob
from typing import Protocol

# Token -> evdev name. Letters and digits are generated; everything else is
# spelled out so a binding string stays readable in gestures.json.
EVDEV_KEYS = {c: f'KEY_{c.upper()}' for c in 'abcdefghijklmnopqrstuvwxyz0123456789'}
EVDEV_KEYS.update({
    'ctrl': 'KEY_LEFTCTRL', 'alt': 'KEY_LEFTALT', 'shift': 'KEY_LEFTSHIFT',
    'super': 'KEY_LEFTMETA', 'altgr': 'KEY_RIGHTALT',
    'enter': 'KEY_ENTER', 'esc': 'KEY_ESC', 'tab': 'KEY_TAB', 'space': 'KEY_SPACE',
    'backspace': 'KEY_BACKSPACE', 'delete': 'KEY_DELETE', 'insert': 'KEY_INSERT',
    'home': 'KEY_HOME', 'end': 'KEY_END', 'pageup': 'KEY_PAGEUP', 'pagedown': 'KEY_PAGEDOWN',
    'up': 'KEY_UP', 'down': 'KEY_DOWN', 'left': 'KEY_LEFT', 'right': 'KEY_RIGHT',
    'minus': 'KEY_MINUS', 'equal': 'KEY_EQUAL', 'comma': 'KEY_COMMA', 'dot': 'KEY_DOT',
    'slash': 'KEY_SLASH', 'semicolon': 'KEY_SEMICOLON', 'grave': 'KEY_GRAVE',
    'volumeup': 'KEY_VOLUMEUP', 'volumedown': 'KEY_VOLUMEDOWN', 'mute': 'KEY_MUTE',
    'playpause': 'KEY_PLAYPAUSE', 'nexttrack': 'KEY_NEXTSONG', 'prevtrack': 'KEY_PREVIOUSSONG',
})
EVDEV_KEYS.update({f'f{n}': f'KEY_F{n}' for n in range(1, 13)})

# Token -> pynput Key attribute. Absent tokens are typed as characters.
PYNPUT_KEYS = {
    'ctrl': 'ctrl', 'alt': 'alt', 'shift': 'shift', 'super': 'cmd', 'altgr': 'alt_gr',
    'enter': 'enter', 'esc': 'esc', 'tab': 'tab', 'space': 'space',
    'backspace': 'backspace', 'delete': 'delete', 'insert': 'insert',
    'home': 'home', 'end': 'end', 'pageup': 'page_up', 'pagedown': 'page_down',
    'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
    'volumeup': 'media_volume_up', 'volumedown': 'media_volume_down',
    'mute': 'media_volume_mute', 'playpause': 'media_play_pause',
    'nexttrack': 'media_next', 'prevtrack': 'media_previous',
}
PYNPUT_KEYS.update({f'f{n}': f'f{n}' for n in range(1, 13)})


class InputBackend(Protocol):
    def move(self, x, y): ...
    def down(self): ...
    def up(self): ...
    def right(self): ...
    def scroll(self, steps): ...
    def close(self): ...

class PynputInput:
    def __init__(self, bounds):
        from pynput.mouse import Controller, Button
        self.mouse, self.button = Controller(), Button
    def move(self, x, y): self.mouse.position = (x, y)
    def down(self): self.mouse.press(self.button.left)
    def up(self): self.mouse.release(self.button.left)
    def right(self): self.mouse.click(self.button.right)
    def scroll(self, steps): self.mouse.scroll(0, steps)
    def close(self): self.up()

class UInputMouse:
    def __init__(self, bounds):
        from evdev import UInput, AbsInfo, ecodes as e
        self.e, self.bounds = e, bounds
        self.device = UInput({e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
            e.EV_REL: [e.REL_WHEEL],
            e.EV_ABS: [(e.ABS_X, AbsInfo(0, 0, 32767, 0, 0, 0)),
                       (e.ABS_Y, AbsInfo(0, 0, 32767, 0, 0, 0))]},
            name='AirMouse virtual pointer', input_props=[e.INPUT_PROP_POINTER])
    def move(self, x, y):
        ox, oy, w, h = self.bounds
        self.device.write(self.e.EV_ABS, self.e.ABS_X, round(max(0, min(1, (x-ox)/max(1,w-1)))*32767))
        self.device.write(self.e.EV_ABS, self.e.ABS_Y, round(max(0, min(1, (y-oy)/max(1,h-1)))*32767))
        self.device.syn()
    def key(self, key, value):
        self.device.write(self.e.EV_KEY, key, value)
        self.device.syn()
    def down(self): self.key(self.e.BTN_LEFT, 1)
    def up(self): self.key(self.e.BTN_LEFT, 0)
    def right(self):
        self.key(self.e.BTN_RIGHT, 1)
        self.key(self.e.BTN_RIGHT, 0)
    def scroll(self, steps):
        self.device.write(self.e.EV_REL, self.e.REL_WHEEL, steps)
        self.device.syn()
    def close(self):
        try: self.up()
        finally: self.device.close()

class UInputKeyboard:
    """Separate virtual device from the pointer: compositors classify a node by
    the capabilities it advertises, and a single device claiming both absolute
    pointer and keyboard gets handled inconsistently across compositors."""
    def __init__(self):
        from evdev import UInput, ecodes as e
        self.e = e
        codes = sorted({e.ecodes[name] for name in EVDEV_KEYS.values() if name in e.ecodes})
        self.device = UInput({e.EV_KEY: codes}, name='AirMouse virtual keyboard')

    def tap(self, tokens):
        """Press modifiers, strike the final key, release in reverse order."""
        codes = [self.e.ecodes[EVDEV_KEYS[t]] for t in tokens]
        for code in codes:
            self.device.write(self.e.EV_KEY, code, 1)
        self.device.syn()
        for code in reversed(codes):
            self.device.write(self.e.EV_KEY, code, 0)
        self.device.syn()

    def close(self):
        self.device.close()


class PynputKeyboard:
    def __init__(self):
        from pynput import keyboard
        self.keyboard = keyboard
        self.controller = keyboard.Controller()

    def _key(self, token):
        named = getattr(self.keyboard.Key, PYNPUT_KEYS.get(token, ''), None)
        return named if named is not None else self.keyboard.KeyCode.from_char(token)

    def tap(self, tokens):
        keys = [self._key(t) for t in tokens]
        for key in keys[:-1]:
            self.controller.press(key)
        try:
            self.controller.press(keys[-1])
            self.controller.release(keys[-1])
        finally:
            for key in reversed(keys[:-1]):
                self.controller.release(key)

    def close(self):
        pass


def create_keyboard():
    return UInputKeyboard() if wayland() else PynputKeyboard()


def wayland():
    return os.environ.get('XDG_SESSION_TYPE') == 'wayland' or bool(os.environ.get('WAYLAND_DISPLAY'))

def create_backend(bounds):
    return UInputMouse(bounds) if wayland() else PynputInput(bounds)

class Hotkeys:
    """F8 toggles; F12 always pauses. Failure disables control, not just hotkeys."""
    def __init__(self, callback, failure):
        self.stop_event = threading.Event()
        self.devices = []
        self.listener = None
        self.thread = None
        if wayland():
            from evdev import InputDevice, ecodes as e
            # evdev 2.0 can return an empty list from list_devices() on some
            # systems even when individual event nodes are accessible. Scan the
            # kernel's stable device-node pattern and let InputDevice validate
            # each candidate instead.
            for path in sorted(glob('/dev/input/event*')):
                try:
                    device = InputDevice(path)
                except OSError:
                    continue
                keys = device.capabilities().get(e.EV_KEY, [])
                if e.KEY_F8 in keys and e.KEY_F12 in keys:
                    self.devices.append(device)
                else:
                    device.close()
            if not self.devices:
                raise RuntimeError('No readable keyboard for global F12 emergency stop. See Linux permissions in README.')
            def listen():
                try:
                    while not self.stop_event.is_set():
                        ready, _, _ = select.select(self.devices, [], [], .1)
                        for device in ready:
                            for event in device.read():
                                if event.type == e.EV_KEY and event.value == 1:
                                    if event.code == e.KEY_F12: callback('stop')
                                    elif event.code == e.KEY_F8: callback('toggle')
                except Exception as exc:
                    if not self.stop_event.is_set(): failure(str(exc))
            self.thread = threading.Thread(target=listen, daemon=True)
            self.thread.start()
        else:
            from pynput import keyboard
            held = set()
            def pressed(key):
                if key in held: return
                held.add(key)
                if key == keyboard.Key.f12: callback('stop')
                elif key == keyboard.Key.f8: callback('toggle')
            self.listener = keyboard.Listener(on_press=pressed, on_release=lambda key: held.discard(key))
            self.listener.start()
            self.listener.wait()

    def healthy(self):
        return self.thread.is_alive() if self.thread else self.listener.is_alive()

    def close(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=1)
        for device in self.devices: device.close()
        if self.listener: self.listener.stop()
