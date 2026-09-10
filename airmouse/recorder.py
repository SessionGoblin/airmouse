"""Record, bind, and manage custom hand-pose templates.

Reads live features the window publishes each frame rather than touching the
camera or the worker, so recording never competes with the vision loop for
frames and control stays paused throughout.
"""
import time
from dataclasses import replace

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


def pose_label(template):
    """One list row. Shared by the full rebuild and the in-place edit, which
    previously formatted rows separately and drifted apart."""
    summary = template.argument or 'unbound'
    if template.role == 'modifier':
        return f'{template.name} — ' + (
            'gates drawing' if template.gates_drawing else summary) + ' [modifier]'
    if template.when:
        return f'{template.name} — {summary} [while "{template.when}"]'
    return f'{template.name} — {summary}'


def stroke_label(stroke):
    suffix = f' [under "{stroke.when}"]' if stroke.when else ''
    return f'{stroke.name} — {stroke.argument or "unbound"}{suffix}'


def capture_layout(dialog, detail_text, ready='Get ready…'):
    """Shared layout for the two capture dialogs.

    The message starts with real text rather than empty: a dialog sizes itself
    when it is shown, and one sized around a blank 20px label clips the first
    message it is later given. The reserved height and minimum width keep the
    window still as well -- these messages change on a 25 ms timer, and a
    dialog that resizes to fit each one jitters continuously.
    """
    dialog.setMinimumWidth(400)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(20, 18, 20, 16)
    layout.setSpacing(12)
    message = QLabel(ready)
    message.setAlignment(Qt.AlignCenter)
    message.setWordWrap(True)
    # Padding here came out of the label's own box and clipped the glyphs; the
    # spacing belongs to the layout instead.
    message.setStyleSheet('font-size: 20px')
    message.setMinimumHeight(62)                # two lines at 20px
    layout.addWidget(message)
    detail = QLabel(detail_text)
    detail.setWordWrap(True)
    detail.setAlignment(Qt.AlignTop)
    detail.setMinimumHeight(44)                 # two lines, so short text does not jump
    layout.addWidget(detail)
    layout.addStretch(1)
    buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    return message, detail


class RecordDialog(QDialog):
    """Countdown, then average the frames held steady into one template."""

    def __init__(self, source, name, parent=None, role='pointer'):
        super().__init__(parent)
        self.setWindowTitle(f'Recording "{name}"')
        self.source = source            # callable -> (hands, timestamp)
        self.name = name
        self.role = role
        self.template = None
        self.samples = []
        self.orientations = []
        self.chiralities = []
        self.built_ins = []
        self.previous = None
        self.started = time.monotonic()
        which = 'modifier' if role == 'modifier' else 'pointer'
        self.message, self.detail = capture_layout(
            self, f'Hold the pose steady with your {which} hand, where the camera can see '
                  'all of it.')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sample)
        self.timer.start(25)

    def sample(self):
        from . import roles
        hands, stamp = self.source()
        hand = roles.by_role(hands or (), self.role)
        features = hand.features if hand else None
        elapsed = time.monotonic() - self.started
        if features is None or stamp is None or time.monotonic()-stamp > FRESH:
            self.message.setText(f'No {self.role} hand visible')
            # A hand that disappears restarts the countdown rather than
            # averaging across the gap.
            self.started = time.monotonic()
            self.samples.clear()
            self.orientations.clear()
            self.chiralities.clear()
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
            self.chiralities.append(features.chirality)
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
        self.template = poses.build(self.name, self.samples, self.orientations,
                                    self.chiralities, role=self.role)
        self.accept()

    def shadows(self, settings):
        # Only the pointer hand runs the built-in gestures, so a modifier pose
        # cannot shadow clicking however tightly it is pinched.
        if self.role == 'modifier':
            return []
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
        # Copy each template, not just the list: sharing the objects meant an
        # edit here mutated what the controller was matching against straight
        # away, and cancelling could not undo it. Poses are immutable tuples,
        # so a shallow dataclass copy is enough.
        self.library = poses.Library([replace(t) for t in library.templates])
        self.settings = settings
        self.source = source
        self.capture = None
        layout = QVBoxLayout(self)
        columns = QHBoxLayout()
        layout.addLayout(columns, 1)

        left = QVBoxLayout()
        columns.addLayout(left)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.select)
        left.addWidget(self.list, 1)
        # Wrap both in lambdas: clicked emits `checked`, and PySide hands it to
        # any slot whose signature can accept an argument, so connecting
        # record() directly passed a bool as `existing`.
        self.record_button = QPushButton('Record new pose…')
        self.record_button.clicked.connect(lambda: self.record(None))
        left.addWidget(self.record_button)
        self.rerecord = QPushButton('Re-record selected')
        self.rerecord.clicked.connect(lambda: self.record(self.current()))
        left.addWidget(self.rerecord)
        self.delete = QPushButton('Delete selected')
        self.delete.clicked.connect(self.remove)
        left.addWidget(self.delete)

        panel = QGroupBox('When this pose is held')
        form = QFormLayout(panel)
        columns.addWidget(panel, 1)
        self.role = QComboBox()
        for key, label in [('pointer', 'Pointer hand'), ('modifier', 'Modifier hand')]:
            self.role.addItem(label, key)
        self.role.setToolTip('Which hand this gesture is read from, and which hand a new '
                             'recording samples. A modifier pose is also a mode: pointer '
                             'poses and drawn strokes can be scoped to it.')
        self.role.currentIndexChanged.connect(self.role_changed)
        form.addRow('Hand', self.role)
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
        self.when = QComboBox()
        self.when.setToolTip('Only fire while this modifier pose is held. Any mode also '
                             'matches with no modifier raised at all.')
        self.when.currentIndexChanged.connect(self.store)
        form.addRow('Only while', self.when)
        self.gates = QCheckBox('Holding this opens stroke drawing')
        self.gates.setToolTip('Replaces the built-in modifier pinch. Strokes can then be scoped '
                              'to this gate, so the same shape means different things under '
                              'different modifier poses.')
        self.gates.toggled.connect(self.store)
        form.addRow(self.gates)
        self.hand = QCheckBox('Only match with the hand it was recorded with')
        self.hand.setToolTip('Off by default, so a pose works with either hand. Turn on to give '
                             'each hand its own gesture. Depends on seeing the palm, so it is '
                             'unreliable with the hand held edge-on.')
        self.hand.toggled.connect(self.store)
        form.addRow(self.hand)
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
            self.list.addItem(pose_label(template))
        if select is not None:
            self.list.setCurrentRow(select)
        elif self.library.templates:
            self.list.setCurrentRow(0)
        self.list.blockSignals(False)
        self.select(self.list.currentRow())

    def select(self, row):
        template = self.current()
        for widget in (self.kind, self.argument, self.app_action, self.tilt, self.hand,
                       self.when, self.gates, self.rerecord, self.delete):
            widget.setEnabled(template is not None)
        # The hand stays live with nothing selected: it is then the hand the
        # next recording will read from.
        self.role.setEnabled(True)
        if template is None:
            self.warning.clear()
            return
        for widget in (self.kind, self.argument, self.app_action, self.tilt, self.hand,
                       self.when, self.gates, self.role):
            widget.blockSignals(True)
        self.role.setCurrentIndex(max(0, self.role.findData(template.role or 'pointer')))
        self.reload_modes(template)
        self.kind.setCurrentIndex(max(0, self.kind.findData(template.action or 'none')))
        self.argument.setText(template.argument if template.action != 'app' else '')
        if template.action == 'app':
            self.app_action.setCurrentIndex(max(0, self.app_action.findData(template.argument)))
        self.tilt.setChecked(template.orientation is not None)
        self.hand.setChecked(template.chirality is not None)
        self.gates.setChecked(template.gates_drawing)
        for widget in (self.kind, self.argument, self.app_action, self.tilt, self.hand,
                       self.when, self.gates, self.role):
            widget.blockSignals(False)
        self.sync_role()
        self.kind_changed()

    def reload_modes(self, template):
        """Fill the scoping list with the modifier poses available as modes."""
        self.when.clear()
        self.when.addItem('Any mode', '')
        for other in self.library.templates:
            if other.role == 'modifier':
                self.when.addItem(f'Holding "{other.name}"', other.name)
        self.when.setCurrentIndex(max(0, self.when.findData(template.when)))

    def sync_role(self):
        """Show only the fields that apply to the selected hand."""
        modifier = self.role.currentData() == 'modifier'
        self.gates.setVisible(modifier)
        self.when.setVisible(not modifier)

    def role_changed(self):
        """Reassign the selected gesture to the other hand.

        Poses are stored mirrored into one chirality, so a recording made with
        one hand matches the other; switching hands needs no re-recording.
        """
        template = self.current()
        if template is not None:
            template.role = self.role.currentData()
        self.sync_role()
        self.store()

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
        template.role = self.role.currentData()
        kind = self.kind.currentData()
        template.action = kind
        template.argument = (self.app_action.currentData() if kind == 'app'
                             else self.argument.text() if kind in ('key', 'shell') else '')
        if self.tilt.isChecked():
            if template.orientation is None:
                template.orientation = getattr(template, 'recorded_orientation', 0.0)
        else:
            template.orientation = None
        template.when = self.when.currentData() or '' if template.role != 'modifier' else ''
        template.gates_drawing = self.gates.isChecked() and template.role == 'modifier'
        if self.hand.isChecked():
            if template.chirality is None:
                template.chirality = getattr(template, 'recorded_chirality', None) or 1
        else:
            template.chirality = None
        self.refresh_row()

    def refresh_row(self):
        row = self.list.currentRow()
        template = self.current()
        if template is None or row < 0:
            return
        self.list.blockSignals(True)
        self.list.item(row).setText(pose_label(template))
        self.list.blockSignals(False)

    def record(self, existing=None):
        if self.capture is not None:
            self.capture.raise_()
            return
        name = existing.name if existing else self.unique_name()
        role = existing.role if existing else self.role.currentData()
        dialog = RecordDialog(self.source, name, self, role=role)
        # Non-modal, like this dialog itself: recording asks you to hold a pose
        # where the camera can see it, which needs the preview underneath.
        dialog.setModal(False)
        self.capture = dialog
        self.setEnabled(False)
        dialog.finished.connect(lambda result: self.recorded(dialog, existing, result))
        dialog.show()
        dialog.raise_()

    def recorded(self, dialog, existing, result):
        self.capture = None
        self.setEnabled(True)
        if not result or dialog.template is None:
            return
        template = dialog.template
        # Remember the recorded tilt so the checkbox can restore it later.
        template.recorded_orientation = template.orientation
        template.recorded_chirality = template.chirality
        if existing is not None:
            template.action, template.argument = existing.action, existing.argument
            template.when, template.gates_drawing = existing.when, existing.gates_drawing
            if existing.orientation is None:
                template.orientation = None
            if existing.chirality is None:
                template.chirality = None
        else:
            template.orientation = None     # tilt-invariant unless asked for
            template.chirality = None       # and works with either hand
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
            template.__dict__.pop('recorded_chirality', None)
        try:
            self.library.save()
        except OSError as exc:
            self.error.setText(f'Could not save gestures: {exc}')
            return
        self.accept()


class DrawDialog(QDialog):
    """Capture one stroke: pinch the modifier hand, trace, then release."""

    def __init__(self, source, name, parent=None, library=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle(f'Drawing "{name}"')
        self.source = source            # callable -> (hands, timestamp)
        self.name = name
        self.library = library
        self.settings = settings
        self.gates = library.gates() if library is not None else set()
        self.gate_open = False
        self.stroke = None
        self.path = []
        self.aspect = 4/3
        self.drawing = False
        self.message, self.detail = capture_layout(
            self, 'Pinch your modifier hand to start drawing, trace the shape with your '
                  'pointer index finger, then open the pinch to finish.')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sample)
        self.timer.start(25)

    def gated(self, modifier):
        """Same hysteresis the controller uses, so recording behaves like use."""
        if self.gates:
            self.gate_open = (modifier.features.pose is not None
                              and self.gate_match(modifier))
            return self.gate_open
        pinch = getattr(self.settings, 'pinch', .32)
        release = getattr(self.settings, 'release', .42)
        self.gate_open = modifier.features.left < (release if self.gate_open else pinch)
        return self.gate_open

    def gate_match(self, modifier):
        found = self.library.match(modifier.features.pose, modifier.features.orientation,
                                   modifier.features.chirality, role='modifier')
        return found is not None and found.name in self.gates

    def sample(self):
        from . import roles, strokes
        hands, stamp = self.source()
        if stamp is None or time.monotonic()-stamp > FRESH:
            self.message.setText('No camera frames')
            return
        pointer = roles.by_role(hands, roles.POINTER)
        modifier = roles.by_role(hands, roles.MODIFIER)
        if modifier is None or modifier.features is None:
            self.message.setText('Show your modifier hand')
            return
        if pointer is None or pointer.features is None:
            self.message.setText('Show your pointer hand')
            return
        gating = self.gated(modifier)
        if gating:
            self.drawing = True
            self.path.append(tuple(pointer.features.point[:2]))
            self.message.setText('Drawing…')
            self.detail.setText(f'{len(self.path)} points')
            return
        if not self.drawing:
            self.message.setText('Pinch the modifier hand to start')
            return
        self.timer.stop()
        points = strokes.canonical(self.path, self.aspect)
        if points is None:
            self.message.setText('Stroke not usable')
            self.detail.setText(strokes.rejection(self.path, self.aspect) or 'Try again.')
            return
        self.stroke = strokes.Stroke(name=self.name, points=tuple(points))
        self.accept()


class StrokeDialog(QDialog):
    """Manage drawn strokes and their bindings."""

    def __init__(self, library, source, parent=None, gates=(), pose_library=None,
                 settings=None):
        super().__init__(parent)
        self.setWindowTitle('Drawn gestures • control remains paused')
        self.resize(620, 420)
        from . import strokes
        self.library = strokes.StrokeLibrary([replace(x) for x in library.strokes])
        self.source = source
        self.gates = tuple(gates)       # modifier poses available as gates
        self.pose_library = pose_library
        self.settings = settings
        self.capture = None
        layout = QVBoxLayout(self)
        columns = QHBoxLayout()
        layout.addLayout(columns, 1)
        left = QVBoxLayout()
        columns.addLayout(left)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.select)
        left.addWidget(self.list, 1)
        self.draw_button = QPushButton('Draw new gesture…')
        self.draw_button.clicked.connect(lambda: self.draw(None))
        left.addWidget(self.draw_button)
        self.redraw = QPushButton('Redraw selected')
        self.redraw.clicked.connect(lambda: self.draw(self.current()))
        left.addWidget(self.redraw)
        self.delete = QPushButton('Delete selected')
        self.delete.clicked.connect(self.remove)
        left.addWidget(self.delete)

        panel = QGroupBox('When this stroke is drawn')
        form = QFormLayout(panel)
        columns.addWidget(panel, 1)
        self.role = QComboBox()
        for key, label in [('pointer', 'Pointer hand'), ('modifier', 'Modifier hand')]:
            self.role.addItem(label, key)
        self.role.setToolTip('Which hand this gesture is read from, and which hand a new '
                             'recording samples. A modifier pose is also a mode: pointer '
                             'poses and drawn strokes can be scoped to it.')
        self.role.currentIndexChanged.connect(self.role_changed)
        form.addRow('Hand', self.role)
        self.kind = QComboBox()
        for key, label in [('none', 'Nothing'), ('key', 'Press a shortcut'),
                           ('app', 'Control AirMouse'), ('shell', 'Run a command')]:
            self.kind.addItem(label, key)
        self.kind.currentIndexChanged.connect(self.kind_changed)
        form.addRow('Action', self.kind)
        self.argument = QLineEdit()
        self.argument.textEdited.connect(self.store)
        form.addRow('Shortcut', self.argument)
        self.app_action = QComboBox()
        for key, label in [('pause', 'Pause control'), ('toggle', 'Pause / resume'),
                           ('recenter', 'Recentre pointer')]:
            self.app_action.addItem(label, key)
        self.app_action.currentIndexChanged.connect(self.store)
        form.addRow('Command', self.app_action)
        self.when = QComboBox()
        self.when.setToolTip('Only match when this modifier pose gated the stroke. Any gate '
                             'also matches whichever gate was held.')
        self.when.currentIndexChanged.connect(self.store)
        form.addRow('Only under', self.when)
        self.rotation = QCheckBox('Match at any orientation')
        self.rotation.setToolTip('Off keeps direction meaningful, so a left swipe and a right '
                                 'swipe stay different gestures. On matches the shape however '
                                 'it is turned, which merges strokes that differ only in '
                                 'direction.')
        self.rotation.toggled.connect(self.store)
        form.addRow(self.rotation)
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
        return self.library.strokes[row] if 0 <= row < len(self.library.strokes) else None

    def refresh(self, select=None):
        self.list.blockSignals(True)
        self.list.clear()
        for stroke in self.library.strokes:
            self.list.addItem(stroke_label(stroke))
        if select is not None:
            self.list.setCurrentRow(select)
        elif self.library.strokes:
            self.list.setCurrentRow(0)
        self.list.blockSignals(False)
        self.select(self.list.currentRow())

    def select(self, row):
        stroke = self.current()
        for widget in (self.kind, self.argument, self.app_action, self.rotation, self.when,
                       self.redraw, self.delete):
            widget.setEnabled(stroke is not None)
        if stroke is None:
            self.warning.clear()
            return
        for widget in (self.kind, self.argument, self.app_action, self.rotation, self.when):
            widget.blockSignals(True)
        self.when.clear()
        self.when.addItem('Any gate', '')
        for gate in self.gates:
            self.when.addItem(f'Under "{gate}"', gate)
        self.when.setCurrentIndex(max(0, self.when.findData(stroke.when)))
        self.kind.setCurrentIndex(max(0, self.kind.findData(stroke.action or 'none')))
        self.argument.setText(stroke.argument if stroke.action != 'app' else '')
        if stroke.action == 'app':
            self.app_action.setCurrentIndex(max(0, self.app_action.findData(stroke.argument)))
        self.rotation.setChecked(stroke.free_rotation)
        for widget in (self.kind, self.argument, self.app_action, self.rotation, self.when):
            widget.blockSignals(False)
        self.when.setVisible(bool(self.gates))
        self.kind_changed()

    def reload_modes(self, template):
        """Fill the scoping list with the modifier poses available as modes."""
        self.when.clear()
        self.when.addItem('Any mode', '')
        for other in self.library.templates:
            if other.role == 'modifier':
                self.when.addItem(f'Holding "{other.name}"', other.name)
        self.when.setCurrentIndex(max(0, self.when.findData(template.when)))

    def sync_role(self):
        """Show only the fields that apply to the selected hand."""
        modifier = self.role.currentData() == 'modifier'
        self.gates.setVisible(modifier)
        self.when.setVisible(not modifier)

    def role_changed(self):
        """Reassign the selected gesture to the other hand.

        Poses are stored mirrored into one chirality, so a recording made with
        one hand matches the other; switching hands needs no re-recording.
        """
        template = self.current()
        if template is not None:
            template.role = self.role.currentData()
        self.sync_role()
        self.store()

    def kind_changed(self):
        kind = self.kind.currentData()
        self.argument.setVisible(kind in ('key', 'shell'))
        self.app_action.setVisible(kind == 'app')
        self.argument.setPlaceholderText('ctrl+alt+t' if kind == 'key' else 'firefox --new-window')
        self.store()

    def store(self):
        stroke = self.current()
        if stroke is None:
            return
        kind = self.kind.currentData()
        stroke.action = kind
        stroke.argument = (self.app_action.currentData() if kind == 'app'
                           else self.argument.text() if kind in ('key', 'shell') else '')
        stroke.free_rotation = self.rotation.isChecked()
        stroke.when = self.when.currentData() or ''
        row = self.list.currentRow()
        if row >= 0:
            self.list.blockSignals(True)
            self.list.item(row).setText(stroke_label(stroke))
            self.list.blockSignals(False)

    def draw(self, existing=None):
        if self.capture is not None:
            self.capture.raise_()
            return
        name = existing.name if existing else self.unique_name()
        dialog = DrawDialog(self.source, name, self, library=self.pose_library,
                            settings=self.settings)
        dialog.setModal(False)
        self.capture = dialog
        self.setEnabled(False)
        dialog.finished.connect(lambda result: self.drawn(dialog, existing, result))
        dialog.show()
        dialog.raise_()

    def drawn(self, dialog, existing, result):
        self.capture = None
        self.setEnabled(True)
        if not result or dialog.stroke is None:
            return
        stroke = dialog.stroke
        if existing is not None:
            stroke.action, stroke.argument = existing.action, existing.argument
            stroke.free_rotation, stroke.when = existing.free_rotation, existing.when
        clash = self.library.conflict(stroke)
        self.library.replace(stroke)
        self.refresh(select=[s.name for s in self.library.strokes].index(stroke.name))
        self.warning.setText(
            f'Scores as high against "{clash.name}" as a real match would — the two cannot be '
            'told apart. Redraw one of them with a clearly different shape.' if clash else '')

    def unique_name(self):
        taken = {s.name for s in self.library.strokes}
        n = 1
        while f'Stroke {n}' in taken:
            n += 1
        return f'Stroke {n}'

    def remove(self):
        stroke = self.current()
        if stroke is None:
            return
        if QMessageBox.question(self, 'Delete gesture', f'Delete "{stroke.name}"?') \
                != QMessageBox.Yes:
            return
        self.library.remove(stroke.name)
        self.refresh()

    def apply(self):
        for stroke in self.library.strokes:
            try:
                stroke.argument = actions.validate(stroke.action, stroke.argument)
            except ValueError as exc:
                self.error.setText(f'{stroke.name}: {exc}')
                return
        try:
            self.library.save()
        except OSError as exc:
            self.error.setText(f'Could not save gestures: {exc}')
            return
        self.accept()
