import argparse
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
        self.controller = Controller(self.settings, (rect.x(), rect.y(), rect.width(), rect.height()))
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
        self.start_button = QPushButton('Start preview')
        self.start_button.clicked.connect(self.start_stop)
        form.addRow(self.start_button)
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
        for key, label in [('left','Pinch click & drag'),('right','Middle pinch right click'),('scroll','Two-finger scroll')]:
            box = QCheckBox(label)
            box.setChecked(getattr(self.settings,key))
            box.toggled.connect(lambda value, k=key: self.setting(k,value))
            form.addRow(box)
        calibration = QPushButton('Calibrate active region & gestures…')
        calibration.clicked.connect(self.calibrate)
        form.addRow(calibration)
        self.debug = QCheckBox('Show tuning diagnostics')
        self.debug.setChecked(debug)
        form.addRow(self.debug)
        form.addRow(QLabel('Point: index finger\nClick / drag: thumb + index\nRight click: thumb + middle\nScroll: index + middle extended,\nring + little finger folded; move up/down'))
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
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(30)

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
        self.status.setText('PAUSED • starting camera and model…')

    def stop(self):
        self.pause()
        if self.worker and not self.worker.close():
            self.platform_status.setText('Camera is still shutting down; control remains disabled.')
            return
        self.worker = None
        self.start_button.setText('Start preview')
        self.camera.setEnabled(True)
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

    def calibrate(self):
        self.pause()
        self.calibrating = True
        try:
            dialog = CalibrationDialog(self.settings,self)
            if dialog.exec():
                self.settings = dialog.settings
                self.controller.settings = self.settings
                for key, box in self.adjusters.items():
                    box.blockSignals(True)
                    box.setValue(getattr(self.settings,key))
                    box.blockSignals(False)
        finally:
            self.calibrating = False
            self.pause()

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
        frame, points, confidence, features, state, fps = item
        import cv2
        h,w = frame.shape[:2]
        margin = self.settings.margin
        cv2.rectangle(frame,(int(w*margin),int(h*margin)),(int(w*(1-margin)),int(h*(1-margin))),(120,220,90),2)
        if points:
            coords = [(int(p[0]*w),int(p[1]*h)) for p in points]
            for a,b in EDGES: cv2.line(frame,coords[a],coords[b],(220,190,60),2)
            for i, point in enumerate(coords):
                cv2.circle(frame,point,4,(70,250,220),-1)
                if self.debug.isChecked(): cv2.putText(frame,str(i),point,cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
            cv2.drawMarker(frame,coords[8],(255,255,255),cv2.MARKER_CROSS,22,2)
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        enabled = self.controller.enabled
        self.resume_button.setEnabled(bool(self.controller.backend and self.keys and self.keys.healthy() and not self.input_error))
        self.resume_button.setText('Pause control' if enabled else 'Enable control')
        tracking = f'{confidence[0]} hand • handedness {confidence[1]:.0%}' if confidence else 'No hand • waiting'
        self.status.setText(f'{"ACTIVE" if enabled else "PAUSED"} • {state} • {tracking} • {fps:.0f} FPS')
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
                text += ' '.join(f'{i}:({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})' for i,p in enumerate(points))+'\n'
            self.diagnostics.setPlainText(text+'\n'.join(self.transitions))

    def closeEvent(self, event):
        self.stop()
        if self.worker:
            event.ignore()
            return
        if self.keys: self.keys.close()
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
    sys.exit(app.exec())
