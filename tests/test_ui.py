import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from airmouse.ui import Window
from airmouse.config import Settings
from airmouse.calibration import CalibrationDialog

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
