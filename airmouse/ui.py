import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QCheckBox, QComboBox, QDoubleSpinBox,
    QPlainTextEdit, QFormLayout, QGroupBox, QMessageBox)
from .config import Settings
from .controller import Controller
from .calibration import CalibrationDialog
from . import actions, perf, poses, roles, strokes

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
         (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

class Window(QMainWindow):
    hotkey = Signal(str)
    hotkey_error = Signal(str)

    def __init__(self, debug=False):
        super().__init__()
        self.setWindowTitle('AirMouse • local hand control')
        self.resize(1040, 760)
        self.settings = Settings.load()
        rect = QApplication.primaryScreen().virtualGeometry()
        screens = [screen.geometry() for screen in QApplication.primaryScreen().virtualSiblings()]
        self.library = poses.Library.load()
        self.stroke_library = strokes.StrokeLibrary.load()
        # App bindings hop to the GUI thread through the existing hotkey signal,
        # which already exists for exactly this: the dispatcher runs on the
        # vision thread and must not touch widgets.
        self.dispatcher = actions.Dispatcher(app=self.receive_app_action)
        self.controller = Controller(self.settings, (rect.x(), rect.y(), rect.width(), rect.height()),
                                     screens=[(r.x(), r.y(), r.width(), r.height()) for r in screens],
                                     library=self.library, dispatcher=self.dispatcher,
                                     stroke_library=self.stroke_library)
        self.last_hands = ([], None)
        self.dialog = None
        self.worker = self.keys = None
        self.calibrating = False
        self.previous_state = ''
        self.transitions = []
        self.input_error = ''
        self.hotkey.connect(self.on_hotkey)
        self.hotkey_error.connect(self.on_hotkey_error)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        heading = QLabel('AirMouse')
        heading.setStyleSheet('font-size: 28px; font-weight: bold')
        layout.addWidget(heading)
        layout.addWidget(QLabel('Private webcam control  •  F8 pause / resume  •  F12 emergency stop'))
        row = QHBoxLayout()
        layout.addLayout(row, 1)
        self.preview = QLabel('Start preview to connect your webcam.\nPointer control starts paused.')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(480, 360)
        self.preview.setStyleSheet('background: #101b27; color: #b7cbdc; border-radius: 10px')
        row.addWidget(self.preview, 1)
        panel = QGroupBox('Control & calibration')
        panel.setMaximumWidth(320)
        form = QFormLayout(panel)
        row.addWidget(panel)
        self.camera = QComboBox()
        devices = sorted(Path('/dev').glob('video*')) if sys.platform == 'linux' else []
        indices = [int(p.name[5:]) for p in devices if p.name[5:].isdigit()] or list(range(4))
        if self.settings.camera not in indices: indices.append(self.settings.camera)
        for index in indices: self.camera.addItem(f'Webcam {index}', index)
        self.camera.setCurrentIndex(self.camera.findData(self.settings.camera))
        form.addRow('Camera', self.camera)
        self.resolution = QComboBox()
        sizes = [(320, 240), (640, 480), (800, 600), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
        selected = (self.settings.camera_width, self.settings.camera_height)
        if selected not in sizes: sizes.append(selected)
        for size in sorted(sizes):
            self.resolution.addItem(f'{size[0]} × {size[1]}', size)
            if size == selected: self.resolution.setCurrentIndex(self.resolution.count()-1)
        self.resolution.setToolTip('Requested capture size. Stop preview to change it; the camera may use a different size.')
        form.addRow('Resolution', self.resolution)
        self.hold_fps = QCheckBox('Hold frame rate in low light')
        self.hold_fps.setChecked(self.settings.hold_fps)
        self.hold_fps.setToolTip(
            'When on, webcams that drop to 15 FPS for brightness are asked to stay at 30 FPS. '
            'The preview may look darker. Stop preview to change this.')
        self.hold_fps.toggled.connect(lambda value: self.setting('hold_fps', value))
        form.addRow(self.hold_fps)
        self.start_button = QPushButton('Start preview')
        self.start_button.clicked.connect(self.start_stop)
        form.addRow(self.start_button)
        from .input import wayland
        if sys.platform == 'linux' and wayland():
            permissions = QPushButton('Set up input permissions…')
            permissions.clicked.connect(lambda: self.check_permissions(force=True))
            form.addRow(permissions)
        self.resume_button = QPushButton('Enable control')
        self.resume_button.setEnabled(False)
        self.resume_button.clicked.connect(self.toggle)
        form.addRow(self.resume_button)
        emergency = QPushButton('Emergency stop · F12')
        emergency.setStyleSheet('background: #a43036; color: white; padding: 10px')
        emergency.clicked.connect(self.pause)
        form.addRow(emergency)
        self.adjusters = {}
        for key, label, lo, hi, step in [('sensitivity','Sensitivity',1,2,.05), ('smoothing','Smoothing (s)',.01,.4,.01)]:
            box = QDoubleSpinBox()
            box.setRange(lo,hi)
            box.setSingleStep(step)
            box.setValue(getattr(self.settings,key))
            box.valueChanged.connect(lambda value, k=key: self.setting(k,value))
            self.adjusters[key] = box
            form.addRow(label, box)
        self.two_hands = QCheckBox('Track a second hand')
        self.two_hands.setChecked(self.settings.two_hands)
        self.two_hands.setToolTip('Adds a modifier hand alongside the pointer. Costs frame rate: '
                                  'the model keeps hunting for a second hand whenever only one is '
                                  'visible. Stop preview to change it.')
        self.two_hands.toggled.connect(lambda value: self.setting('two_hands', value))
        form.addRow(self.two_hands)
        self.pointer_side = QComboBox()
        for key, label in [('right', 'Right hand points'), ('left', 'Left hand points'),
                           ('auto', 'Automatic')]:
            self.pointer_side.addItem(label, key)
        self.pointer_side.setCurrentIndex(self.pointer_side.findData(self.settings.pointer_side))
        self.pointer_side.setToolTip('Right or Left pins the pointer to that side of the '
                                     'mirrored preview and never drifts; crossing your hands '
                                     'swaps them. Automatic follows each hand through a '
                                     'crossing, but can settle the wrong way round.')
        self.pointer_side.currentIndexChanged.connect(
            lambda: self.setting('pointer_side', self.pointer_side.currentData()))
        form.addRow('Pointer', self.pointer_side)
        for key, label in [('left','Pinch click & drag'),('right','Middle pinch right click'),('scroll','Two-finger scroll')]:
            box = QCheckBox(label)
            box.setChecked(getattr(self.settings,key))
            box.toggled.connect(lambda value, k=key: self.setting(k,value))
            form.addRow(box)
        calibration = QPushButton('Calibrate active region & gestures…')
        calibration.clicked.connect(self.calibrate)
        form.addRow(calibration)
        custom = QPushButton('Custom gestures…')
        custom.clicked.connect(self.edit_gestures)
        form.addRow(custom)
        drawn = QPushButton('Drawn gestures…')
        drawn.clicked.connect(self.edit_strokes)
        form.addRow(drawn)
        self.show_preview = QCheckBox('Show preview')
        self.show_preview.setChecked(True)
        self.show_preview.setToolTip('Turn off to stop all preview rendering; hand tracking and control keep running.')
        self.show_preview.toggled.connect(self.on_preview_toggle)
        form.addRow(self.show_preview)
        self.debug = QCheckBox('Show tuning diagnostics')
        self.debug.setChecked(debug)
        form.addRow(self.debug)
        form.addRow(QLabel('Draw: pinch the modifier hand, trace with the pointer hand\n'
                           'Point: index finger\nClick / drag: thumb + index\nRight click: thumb + middle\nScroll: index + middle extended,\nring + little finger folded; move up/down'))
        self.status = QLabel('PAUSED • camera stopped')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.platform_status = QLabel('Input is checked when preview starts.')
        self.platform_status.setWordWrap(True)
        layout.addWidget(self.platform_status)
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setMaximumHeight(175)
        self.diagnostics.setVisible(debug)
        self.debug.toggled.connect(self.diagnostics.setVisible)
        layout.addWidget(self.diagnostics)
        # Local shortcuts supplement, but never replace, required global hotkeys.
        QShortcut(QKeySequence('Escape'), self, activated=self.pause)
        # The 30 ms timer keeps safety checks and status responsive; the preview
        # image is redrawn at most every preview_interval (~17 FPS) so vision
        # keeps the main thread almost to itself.
        self.preview_interval = 1/20
        self.last_preview = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(30)
        self.resolution.currentIndexChanged.connect(self.set_resolution)

    def set_resolution(self):
        self.pause()
        width, height = self.resolution.currentData()
        self.settings = replace(self.settings, camera_width=width, camera_height=height)
        self.controller.settings = self.settings
        try: self.settings.save()
        except OSError as exc: self.platform_status.setText(f'Could not save settings: {exc}')

    def check_permissions(self, force=False):
        from .input import wayland
        if sys.platform != 'linux' or not wayland(): return
        from .permissions import PermissionDialog, access_ready
        if access_ready():
            if force:
                self.pause()
                self.setup_input()
            return
        self.pause()
        self.calibrating = True
        try:
            if PermissionDialog(self).exec():
                self.setup_input()
            else:
                self.platform_status.setText('Preview only • input permission setup was skipped. Use Set up input permissions to retry.')
        finally:
            self.calibrating = False
            self.pause()

    def setting(self, key, value):
        self.pause()
        self.settings = replace(self.settings, **{key:value})
        self.controller.settings = self.settings
        try: self.settings.save()
        except OSError as exc: self.platform_status.setText(f'Could not save settings: {exc}')

    def setup_input(self):
        from .input import Hotkeys, create_backend, wayland
        if self.controller.backend and self.keys and self.keys.healthy(): return
        if self.keys:
            self.keys.close()
            self.keys = None
        if self.controller.backend:
            self.controller.close()
        try:
            self.keys = Hotkeys(self.receive_hotkey, self.receive_hotkey_failure)
            self.controller.backend = create_backend(self.controller.mapper.bounds)
            self.input_error = ''
            self.platform_status.setText(('Wayland uinput' if wayland() else 'Native / X11') + ' input ready • global F8 / F12 active')
        except Exception as exc:
            if self.keys: self.keys.close()
            self.keys = None
            self.input_error = str(exc)
            self.platform_status.setText('Preview only: ' + self.input_error)

    def receive_hotkey_failure(self, error):
        self.controller.pause()
        self.hotkey_error.emit(error)

    def receive_hotkey(self, action):
        if action == 'stop': self.controller.pause()
        self.hotkey.emit(action)

    def receive_app_action(self, action):
        """Called from the vision thread when a pose is bound to app control."""
        if action == 'pause':
            self.controller.pause()
            self.hotkey.emit('stop')
        elif action == 'toggle':
            self.hotkey.emit('toggle')

    def on_hotkey_error(self, error):
        self.pause()
        self.input_error = error
        self.resume_button.setEnabled(False)
        self.platform_status.setText('Global hotkey unavailable: ' + error)

    def on_hotkey(self, action):
        if action == 'stop': self.pause()
        elif not self.calibrating: self.toggle()

    def start_stop(self):
        if self.worker:
            self.stop()
            return
        self.settings = replace(self.settings, camera=self.camera.currentData())
        self.controller.settings = self.settings
        self.setup_input()
        from .worker import VisionWorker
        self.worker = VisionWorker(self.controller)
        self.start_button.setText('Stop camera')
        self.camera.setEnabled(False)
        self.resolution.setEnabled(False)
        self.hold_fps.setEnabled(False)
        self.two_hands.setEnabled(False)
        self.status.setText('PAUSED • starting camera and model…')

    def close_dialog(self):
        if self.dialog is not None:
            self.dialog.reject()
            self.dialog = None
            self.calibrating = False

    def stop(self):
        self.close_dialog()
        self.pause()
        if self.worker and not self.worker.close():
            self.platform_status.setText('Camera is still shutting down; control remains disabled.')
            return
        self.worker = None
        self.start_button.setText('Start preview')
        self.camera.setEnabled(True)
        self.resolution.setEnabled(True)
        self.hold_fps.setEnabled(True)
        self.two_hands.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.status.setText('PAUSED • camera stopped')

    def pause(self):
        self.controller.pause()
        self.resume_button.setText('Enable control')
        self.status.setText('PAUSED • pointer control disabled')

    def toggle(self):
        if self.controller.enabled:
            self.pause()
        elif (not self.calibrating and self.worker and not self.worker.error and self.keys
              and self.keys.healthy() and time.monotonic()-self.worker.last_frame < .3):
            if self.controller.resume(): self.resume_button.setText('Pause control')

    def open_dialog(self, dialog, applied):
        """Show a gesture dialog without blocking the main window.

        These dialogs ask you to hold a pose or draw a shape, which needs the
        live preview they would otherwise cover. Modal dialogs kept the timer
        running but hid the one thing you need to see, so they are shown
        non-modally and control stays paused for as long as one is open.
        """
        if self.dialog is not None:
            self.dialog.raise_()
            self.dialog.activateWindow()
            return
        self.pause()
        self.calibrating = True
        self.dialog = dialog
        dialog.setModal(False)
        dialog.setAttribute(Qt.WA_DeleteOnClose, False)

        def finished(result):
            self.dialog = None
            self.calibrating = False
            if result:
                applied(dialog)
            self.pause()
        dialog.finished.connect(finished)
        dialog.show()
        dialog.raise_()

    def edit_gestures(self):
        from .recorder import GestureDialog

        def applied(dialog):
            self.library = dialog.library
            # The machine holds the library directly, and reset() re-runs
            # __init__ with it, so swapping the reference is enough.
            self.controller.machine.library = self.library
        self.open_dialog(GestureDialog(self.library, self.settings,
                                       lambda: self.last_hands, self), applied)

    def edit_strokes(self):
        from .recorder import StrokeDialog

        def applied(dialog):
            self.stroke_library = dialog.library
            self.controller.stroke_library = self.stroke_library
        self.open_dialog(StrokeDialog(self.stroke_library, lambda: self.last_hands, self,
                                      gates=sorted(self.library.gates()),
                                      pose_library=self.library,
                                      settings=self.settings), applied)

    def calibrate(self):
        def applied(dialog):
            self.settings = dialog.settings
            self.controller.settings = self.settings
            for key, box in self.adjusters.items():
                box.blockSignals(True)
                box.setValue(getattr(self.settings,key))
                box.blockSignals(False)
        self.open_dialog(CalibrationDialog(self.settings, self), applied)

    def on_preview_toggle(self, on):
        if not on:
            self.preview.setText('Preview hidden • hand tracking continues')

    def render_preview(self, frame, hands, w, h):
        import cv2
        start = time.perf_counter()
        # Shrink to the display size before any per-pixel work, so colour
        # conversion and the pixmap only ever touch the ~480 px preview. This
        # bounds cost and memory bandwidth regardless of capture resolution: a
        # 4K frame no longer feeds a full-frame convert + copy + smooth scale
        # onto the main thread where it would compete with inference.
        label = self.preview.size()
        scale = min(max(1, label.width())/w, max(1, label.height())/h)
        dw, dh = max(1, round(w*scale)), max(1, round(h*scale))
        view = cv2.resize(frame, (dw, dh),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        margin = self.settings.margin
        cv2.rectangle(view,(int(dw*margin),int(dh*margin)),(int(dw*(1-margin)),int(dh*(1-margin))),(120,220,90),2)
        for hand in hands or ():
            if not hand.points: continue
            # Colour by role, so a role swapping between hands is visible at a
            # glance rather than only as the cursor jumping.
            pointer = hand.role == roles.POINTER
            bone = (220,190,60) if pointer else (150,120,210)
            joint = (70,250,220) if pointer else (200,170,255)
            coords = [(int(p[0]*dw),int(p[1]*dh)) for p in hand.points]
            for a,b in EDGES: cv2.line(view,coords[a],coords[b],bone,2)
            for i, point in enumerate(coords):
                cv2.circle(view,point,4,joint,-1)
                if self.debug.isChecked(): cv2.putText(view,str(i),point,cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
            if pointer:
                cv2.drawMarker(view,coords[8],(255,255,255),cv2.MARKER_CROSS,22,2)
            if hand.role:
                cv2.putText(view,hand.role,(coords[0][0]-20,coords[0][1]+18),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,bone,1)
        # QImage borrows rgb's buffer and QPixmap.fromImage copies it out
        # synchronously; rgb stays referenced until this method returns, so the
        # defensive full-frame QImage.copy() the old path used is unnecessary.
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, dw, dh, rgb.strides[0], QImage.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(image))
        if perf.PROFILER.enabled:
            perf.PROFILER.record('preview', (time.perf_counter()-start)*1000)

    def refresh(self):
        if not self.worker: return
        if self.keys and not self.keys.healthy():
            self.on_hotkey_error('Keyboard listener stopped; restart preview to reconnect.')
        age = time.monotonic()-self.worker.last_frame
        if self.worker.error:
            error = self.worker.error
            self.stop()
            self.platform_status.setText(error)
            return
        if age > .3 and self.controller.enabled: self.pause()
        item = self.worker.take()
        if not item: return
        frame, hands, features, state, fps, captured = item
        pointer = roles.by_role(hands, roles.POINTER)
        points = pointer.points if pointer else None
        self.last_hands = (hands, time.monotonic())
        h,w = frame.shape[:2]
        if self.show_preview.isChecked() and time.monotonic()-self.last_preview >= self.preview_interval:
            self.last_preview = time.monotonic()
            self.render_preview(frame, hands, w, h)
        enabled = self.controller.enabled
        self.resume_button.setEnabled(bool(self.controller.backend and self.keys and self.keys.healthy() and not self.input_error))
        self.resume_button.setText('Pause control' if enabled else 'Enable control')
        if pointer:
            tracking = f'{len(hands)} hand{"s" if len(hands) != 1 else ""}'
            if pointer.label:
                tracking += f' • pointer reads {pointer.label} {pointer.score:.0%}'
        else:
            tracking = 'No hand • waiting'
        requested = (self.settings.camera_width, self.settings.camera_height)
        resolution = f'{captured[0]} × {captured[1]}'
        if captured != requested:
            resolution += f' (requested {requested[0]} × {requested[1]})'
        mode = f' • mode {self.controller.mode}' if self.controller.mode else ''
        self.status.setText(f'{"ACTIVE" if enabled else "PAUSED"} • {state}{mode} • {tracking} • {resolution} • {fps:.0f} FPS')
        if state != self.previous_state:
            self.transitions.append(f'{time.strftime("%H:%M:%S")} {self.previous_state or "start"} → {state}')
            self.transitions = self.transitions[-6:]
            self.previous_state = state
        if self.debug.isChecked():
            text = f'FPS {fps:.1f} | cursor target {self.controller.target}\n'
            if features:
                text += f'Pinch / palm: index {features.left:.3f}, middle {features.right:.3f}; scroll pose {features.scroll}\n'
                from .gestures import joint_angle
                text += f'Index PIP angle {joint_angle(points, 5, 6, 8):.1f}°; middle PIP angle {joint_angle(points, 9, 10, 12):.1f}°\n'
                text += self.pose_diagnostics(features)
                name, rating = self.controller.last_stroke
                if name:
                    verdict = 'matched' if rating >= strokes.THRESHOLD else 'below threshold'
                    text += f'Last stroke: nearest "{name}" {rating:.3f} -> {verdict}\n'
                text += ' '.join(f'{i}:({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})' for i,p in enumerate(points))+'\n'
            self.diagnostics.setPlainText(text+'\n'.join(self.transitions))

    def pose_diagnostics(self, features):
        """Why a custom pose did or did not fire, in the units it is judged in.

        Reports the nearest template with its distance and accept radius, since
        a pose that never fires looks identical to one that is not recognized at
        all. Arming is included because no gesture of any kind is considered
        until a neutral open hand has been held for the arming dwell.
        """
        if not self.library.templates:
            return 'Custom poses: none recorded\n'
        if not self.settings.custom:
            return 'Custom poses: disabled in settings\n'
        template, gap = self.library.nearest(features.pose)
        if template is None:
            return 'Custom poses: no pose from this frame\n'
        verdict = 'MATCH' if gap <= template.threshold else 'too far'
        if (gap <= template.threshold and template.orientation is not None
                and features.orientation is not None):
            from . import poses as _poses
            off = abs(_poses.angle_delta(features.orientation, template.orientation))
            if off > template.tolerance:
                verdict = f'shape ok, tilt off by {math.degrees(off):.0f}°'
        if (gap <= template.threshold and template.chirality is not None
                and features.chirality is not None
                and features.chirality != template.chirality):
            verdict = 'shape ok, wrong hand'
        armed = 'armed' if self.controller.machine.armed else 'NOT armed (hold an open hand)'
        mode = self.controller.mode
        from . import poses as _poses
        turn = _poses.winding(features.pose)
        hand = ('either' if template.chirality is None else
                f'locked, this hand {features.chirality}')
        return (f'Nearest pose "{template.name}" {gap:.3f} / {template.threshold:.3f} '
                f'-> {verdict} • {armed}\n'
                f'Palm winding {turn:+.2f} (ambiguous under {_poses.AMBIGUOUS}) • hand {hand}\n'
                f'Modifier mode: {mode or "none"}\n')

    def closeEvent(self, event):
        self.close_dialog()
        self.stop()
        if self.worker:
            event.ignore()
            return
        if self.keys: self.keys.close()
        self.dispatcher.close()
        self.controller.close()
        event.accept()

def main():
    parser = argparse.ArgumentParser(description='Local-only webcam hand mouse')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--download-model', action='store_true')
    args = parser.parse_args()
    if args.download_model:
        from .tracking import download_model
        print(download_model())
        return
    app = QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    window = Window(args.debug)
    window.show()
    QTimer.singleShot(0, window.check_permissions)
    sys.exit(app.exec())
