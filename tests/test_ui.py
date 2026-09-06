import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from airmouse.ui import Window
from airmouse.config import Settings
from airmouse.calibration import CalibrationDialog
from airmouse import input as input_module

app = QApplication.instance() or QApplication([])

def test_safe_start_and_calibration_hotkey_lockout(monkeypatch, tmp_path):
    monkeypatch.setattr('airmouse.config.CONFIG_PATH', tmp_path/'settings.json')
    w = Window()
    assert not w.controller.enabled
    assert not w.resume_button.isEnabled()
    w.calibrating = True
    w.on_hotkey('toggle')
    assert not w.controller.enabled
    w.setting('sensitivity',1.25)
    assert Settings.load().sensitivity == 1.25
    w.close()

def test_calibration_rejects_invalid_hysteresis(monkeypatch,tmp_path):
    monkeypatch.setattr('airmouse.config.CONFIG_PATH',tmp_path/'settings.json')
    dialog = CalibrationDialog(Settings())
    dialog.fields['pinch'].setValue(.5)
    dialog.fields['release'].setValue(.2)
    dialog.apply()
    assert 'exceed' in dialog.error.text()
    assert not (tmp_path/'settings.json').exists()
    dialog.close()

def test_wayland_hotkeys_scan_event_nodes_when_evdev_discovery_is_empty(monkeypatch):
    class FakeDevice:
        def __init__(self, path):
            self.path = path
        def capabilities(self):
            from evdev import ecodes as e
            return {e.EV_KEY: [e.KEY_F8, e.KEY_F12]}
        def close(self):
            pass

    monkeypatch.setattr(input_module, 'wayland', lambda: True)
    monkeypatch.setattr(input_module, 'glob', lambda pattern: ['/dev/input/event5'])
    monkeypatch.setattr('evdev.InputDevice', FakeDevice)
    monkeypatch.setattr(input_module.threading.Thread, 'start', lambda self: None)
    hotkeys = input_module.Hotkeys(lambda action: None, lambda error: None)
    assert [device.path for device in hotkeys.devices] == ['/dev/input/event5']
