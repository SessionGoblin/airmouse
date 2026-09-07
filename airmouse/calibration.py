from dataclasses import replace
from PySide6.QtWidgets import QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox, QLabel

class CalibrationDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Calibration • control remains paused')
        self.settings = replace(settings)
        form = QFormLayout(self)
        form.addRow(QLabel('Use the mirrored preview to keep your hand inside the green region.\nApply saves settings; explicitly resume afterward.'))
        self.fields = {}
        fields = [('margin', 'Active region margin', .05, .4, .01),
                  ('sensitivity', 'Sensitivity', 1, 2, .05),
                  ('smoothing', 'Smoothing time (seconds)', .01, .4, .01),
                  ('deadzone', 'Cursor dead zone (pixels)', 0, 20, .5),
                  ('drag_deadzone', 'Drag start distance (pixels)', 2, 40, 1),
                  ('pinch', 'Pinch / palm ratio', .1, .5, .01),
                  ('release', 'Release / palm ratio', .15, .8, .01),
                  ('debounce', 'Pinch confirmation (seconds)', .03, .5, .01),
                  ('cooldown', 'Action cooldown (seconds)', .1, 2, .05),
                  ('dwell', 'Arm neutral dwell (seconds)', .2, 2, .05)]
        for key, title, lo, hi, step in fields:
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(step)
            box.setValue(getattr(settings, key))
            self.fields[key] = box
            form.addRow(title, box)
        self.error = QLabel()
        form.addRow(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def apply(self):
        try:
            for key, box in self.fields.items(): setattr(self.settings, key, box.value())
            self.settings.save()
        except (ValueError, OSError) as exc:
            self.error.setText(str(exc))
            return
        self.accept()
