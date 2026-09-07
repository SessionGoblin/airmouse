"""Record, bind, and manage custom hand-pose templates.

Reads live features the window publishes each frame rather than touching the
camera or the worker, so recording never competes with the vision loop for
frames and control stays paused throughout.
"""
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QListWidget, QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox,
    QDialogButtonBox, QGroupBox, QMessageBox)

from . import actions, poses

COUNTDOWN = 3          # seconds of "get into the pose" before sampling starts
SAMPLE_SECONDS = 1.2
MIN_SAMPLES = 8
# Frames that move more than this from the previous one are dropped: a template
# averaged over a hand still settling matches nothing later.
STABLE_MOVE = .05
FRESH = .3             # features older than this mean the camera stopped


class RecordDialog(QDialog):
    """Countdown, then average the frames held steady into one template."""

    def __init__(self, source, name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Recording "{name}"')
        self.source = source            # callable -> (features, timestamp)
        self.name = name
        self.template = None
        self.samples = []
        self.orientations = []
        self.built_ins = []
        self.previous = None
        self.started = time.monotonic()
        layout = QVBoxLayout(self)
        self.message = QLabel()
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setStyleSheet('font-size: 20px; padding: 18px')
        layout.addWidget(self.message)
        self.detail = QLabel('Hold the pose steady where the camera can see your whole hand.')
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sample)
        self.timer.start(25)

    def sample(self):
        features, stamp = self.source()
        elapsed = time.monotonic() - self.started
        if features is None or stamp is None or time.monotonic()-stamp > FRESH:
            self.message.setText('No hand visible')
            # A hand that disappears restarts the countdown rather than
            # averaging across the gap.
            self.started = time.monotonic()
            self.samples.clear()
            self.orientations.clear()
            self.previous = None
            return
        if elapsed < COUNTDOWN:
            self.message.setText(f'Hold the pose… {COUNTDOWN-int(elapsed)}')
            return
        if features.pose is None:
            return
        self.message.setText('Recording…')
        # Drop frames while the hand is still settling.
        if self.previous is None or poses.distance(features.pose, self.previous) < STABLE_MOVE:
            self.samples.append(features.pose)
            self.orientations.append(features.orientation)
            self.built_ins.append((features.left, features.right, features.scroll))
        self.previous = features.pose
        self.detail.setText(f'{len(self.samples)} steady frames captured')
        if elapsed >= COUNTDOWN + SAMPLE_SECONDS:
            self.finish()

    def finish(self):
        self.timer.stop()
        if len(self.samples) < MIN_SAMPLES:
            self.message.setText('Pose was too unsteady')
            self.detail.setText(f'Only {len(self.samples)} usable frames. Close and try again, '
                                'holding the pose still under even lighting.')
            return
        self.template = poses.build(self.name, self.samples, self.orientations)
        self.accept()

    def shadows(self, settings):
        """Built-in gestures this recorded pose would also trigger.

        Checked against the pose's own measured pinch ratios rather than by
        comparing templates, because the built-ins are distance thresholds, not
        poses. A closed fist genuinely reads as a click, so this warns instead
        of refusing: templates are matched first, and the user should know the
        pose will shadow clicking while held.
        """
        if not self.built_ins:
            return []
        count = len(self.built_ins)
        left = sum(b[0] for b in self.built_ins)/count
        right = sum(b[1] for b in self.built_ins)/count
        scroll = sum(1 for b in self.built_ins if b[2]) > count/2
        found = []
        if settings.left and left < settings.pinch:
            found.append('pinch click')
        if settings.right and right < settings.pinch:
            found.append('right click')
        if settings.scroll and scroll:
            found.append('two-finger scroll')
        return found


class GestureDialog(QDialog):
    """Manage the template library and what each template does."""

    def __init__(self, library, settings, source, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Custom gestures • control remains paused')
        self.resize(620, 420)
        self.library = poses.Library(list(library.templates))
        self.settings = settings
        self.source = source
        layout = QVBoxLayout(self)
        columns = QHBoxLayout()
        layout.addLayout(columns, 1)

        left = QVBoxLayout()
        columns.addLayout(left)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.select)
        left.addWidget(self.list, 1)
        record = QPushButton('Record new pose…')
        record.clicked.connect(self.record)
        left.addWidget(record)
        self.rerecord = QPushButton('Re-record selected')
        self.rerecord.clicked.connect(lambda: self.record(self.current()))
        left.addWidget(self.rerecord)
        self.delete = QPushButton('Delete selected')
        self.delete.clicked.connect(self.remove)
        left.addWidget(self.delete)

        panel = QGroupBox('When this pose is held')
        form = QFormLayout(panel)
        columns.addWidget(panel, 1)
        self.kind = QComboBox()
        for key, label in [('none', 'Nothing'), ('key', 'Press a shortcut'),
                           ('app', 'Control AirMouse'), ('shell', 'Run a command')]:
            self.kind.addItem(label, key)
        self.kind.currentIndexChanged.connect(self.kind_changed)
        form.addRow('Action', self.kind)
        self.argument = QLineEdit()
        self.argument.setPlaceholderText('ctrl+alt+t')
        self.argument.textEdited.connect(self.store)
        form.addRow('Shortcut', self.argument)
        self.app_action = QComboBox()
        for key, label in [('pause', 'Pause control'), ('toggle', 'Pause / resume'),
                           ('recenter', 'Recentre pointer')]:
            self.app_action.addItem(label, key)
        self.app_action.currentIndexChanged.connect(self.store)
        form.addRow('Command', self.app_action)
        self.tilt = QCheckBox('Only match at the tilt it was recorded at')
        self.tilt.setToolTip('Needed to tell apart poses that are the same shape rotated, '
                             'such as thumbs up and thumbs down.')
        self.tilt.toggled.connect(self.store)
        form.addRow(self.tilt)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet('color: #d08a30')
        form.addRow(self.warning)

        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet('color: #c05055')
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def current(self):
        row = self.list.currentRow()
        return self.library.templates[row] if 0 <= row < len(self.library.templates) else None

    def refresh(self, select=None):
        self.list.blockSignals(True)
        self.list.clear()
        for template in self.library.templates:
            summary = {'key': template.argument, 'shell': template.argument,
                       'app': template.argument}.get(template.action, 'unbound')
            self.list.addItem(f'{template.name} — {summary}')
        if select is not None:
            self.list.setCurrentRow(select)
        elif self.library.templates:
            self.list.setCurrentRow(0)
        self.list.blockSignals(False)
        self.select(self.list.currentRow())

    def select(self, row):
        template = self.current()
        for widget in (self.kind, self.argument, self.app_action, self.tilt,
                       self.rerecord, self.delete):
            widget.setEnabled(template is not None)
        if template is None:
            self.warning.clear()
            return
        for widget in (self.kind, self.argument, self.app_action, self.tilt):
            widget.blockSignals(True)
        self.kind.setCurrentIndex(max(0, self.kind.findData(template.action or 'none')))
        self.argument.setText(template.argument if template.action != 'app' else '')
        if template.action == 'app':
            self.app_action.setCurrentIndex(max(0, self.app_action.findData(template.argument)))
        self.tilt.setChecked(template.orientation is not None)
        for widget in (self.kind, self.argument, self.app_action, self.tilt):
            widget.blockSignals(False)
        self.kind_changed()

    def kind_changed(self):
        kind = self.kind.currentData()
        self.argument.setVisible(kind in ('key', 'shell'))
        self.app_action.setVisible(kind == 'app')
        self.argument.setPlaceholderText('ctrl+alt+t' if kind == 'key' else 'firefox --new-window')
        self.store()

    def store(self):
        """Write the editor back onto the selected template."""
        template = self.current()
        if template is None:
            return
        kind = self.kind.currentData()
        template.action = kind
        template.argument = (self.app_action.currentData() if kind == 'app'
                             else self.argument.text() if kind in ('key', 'shell') else '')
        if self.tilt.isChecked():
            if template.orientation is None:
                template.orientation = getattr(template, 'recorded_orientation', 0.0)
        else:
            template.orientation = None
        self.refresh_row()

    def refresh_row(self):
        row = self.list.currentRow()
        template = self.current()
        if template is None or row < 0:
            return
        summary = template.argument or 'unbound'
        self.list.blockSignals(True)
        self.list.item(row).setText(f'{template.name} — {summary}')
        self.list.blockSignals(False)

    def record(self, existing=None):
        name = existing.name if existing else self.unique_name()
        dialog = RecordDialog(self.source, name, self)
        if not dialog.exec() or dialog.template is None:
            return
        template = dialog.template
        # Remember the recorded tilt so the checkbox can restore it later.
        template.recorded_orientation = template.orientation
        if existing is not None:
            template.action, template.argument = existing.action, existing.argument
            if existing.orientation is None:
                template.orientation = None
        else:
            template.orientation = None     # tilt-invariant unless asked for
        clash = self.library.conflict(template)
        notes = dialog.shadows(self.settings)
        self.library.replace(template)
        self.refresh(select=[t.name for t in self.library.templates].index(template.name))
        messages = []
        if clash is not None:
            messages.append(f'Too close to "{clash.name}" — whichever is nearer wins every '
                            'frame, so one of them will never fire. Re-record one of them.')
        if notes:
            messages.append('This pose also reads as ' + ' and '.join(notes) +
                            ', which it will shadow while held.')
        self.warning.setText('\n'.join(messages))

    def unique_name(self):
        taken = {t.name for t in self.library.templates}
        n = 1
        while f'Gesture {n}' in taken:
            n += 1
        return f'Gesture {n}'

    def remove(self):
        template = self.current()
        if template is None:
            return
        if QMessageBox.question(self, 'Delete gesture', f'Delete "{template.name}"?') \
                != QMessageBox.Yes:
            return
        self.library.remove(template.name)
        self.refresh()

    def apply(self):
        for template in self.library.templates:
            try:
                template.argument = actions.validate(template.action, template.argument)
            except ValueError as exc:
                self.error.setText(f'{template.name}: {exc}')
                return
        for template in self.library.templates:
            # Not persisted; only the dialog uses it.
            template.__dict__.pop('recorded_orientation', None)
        try:
            self.library.save()
        except OSError as exc:
            self.error.setText(f'Could not save gestures: {exc}')
            return
        self.accept()
