"""Desktop adapters and global hotkeys. No gesture knowledge lives here."""
import os
import select
import threading
from typing import Protocol

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
            from evdev import InputDevice, list_devices, ecodes as e
            for path in list_devices():
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
