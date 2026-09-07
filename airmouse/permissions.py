"""Confirmed, asynchronous Linux Wayland input permission setup."""
import os
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton

from .permission_helper import keyboards


def access_ready():
    return (os.access('/dev/uinput', os.R_OK | os.W_OK)
            and any(os.access(path, os.R_OK) for path, _ in keyboards()))


class PermissionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Set up desktop input')
        self.setMinimumWidth(480)
        self.process = None
        layout = QVBoxLayout(self)
        explanation = QLabel(
            'AirMouse needs permission to create a virtual mouse and read a physical '
            'keyboard for F8 and the F12 emergency stop.\n\n'
            'After you confirm, Linux will ask for administrator authentication. '
            'Access is granted to your user for the virtual mouse and the selected '
            'keyboard only. Keyboard access allows other programs running as your '
            'user to read that keyboard too.\n\n'
            'Access can remain after AirMouse closes, until reboot or device '
            'reconnection. Pointer control will stay paused.')
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.keyboard = QComboBox()
        for path, name in keyboards():
            self.keyboard.addItem(f'{name} ({path})', path)
        layout.addWidget(self.keyboard)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.allow = QPushButton('Confirm and grant access…')
        self.allow.clicked.connect(self.grant)
        layout.addWidget(self.allow)
        self.skip = QPushButton('Continue in preview only')
        self.skip.clicked.connect(self.reject)
        layout.addWidget(self.skip)
        required = ['/usr/bin/pkexec', '/usr/bin/python3', '/usr/bin/setfacl', '/usr/sbin/modprobe']
        missing = [p for p in required if not os.access(p, os.X_OK)]
        if missing:
            self.message.setText('Automatic setup is unavailable. Install the system packages '
                                 'providing pkexec, python3, acl and kmod, then retry. Missing: ' + ', '.join(missing))
            self.allow.setEnabled(False)
        elif not self.keyboard.count():
            self.message.setText('No physical keyboard with F8 and F12 was found. Connect one and reopen setup.')
            self.allow.setEnabled(False)

    def grant(self):
        self.allow.setEnabled(False)
        self.keyboard.setEnabled(False)
        self.skip.setEnabled(False)
        self.message.setText('Waiting for administrator authentication…')
        self.process = QProcess(self)
        self.process.finished.connect(self.finished_grant)
        self.process.errorOccurred.connect(self.failed_start)
        self.process.start('/usr/bin/pkexec', ['/usr/bin/python3', '-I',
                           str(Path(__file__).with_name('permission_helper.py').resolve()),
                           self.keyboard.currentData()])

    def failed_start(self, error):
        if error == QProcess.FailedToStart:
            self.finish_error('Could not start administrator authentication. You can continue in preview only.')

    def finished_grant(self, code, status):
        if code == 0 and status == QProcess.NormalExit and access_ready():
            self.accept()
            return
        detail = bytes(self.process.readAllStandardError()).decode(errors='replace').strip()
        self.finish_error('Permission setup was cancelled or failed. Some access may already have been granted. '
                          'You can retry or continue in preview only.\n' + detail)

    def finish_error(self, text):
        self.message.setText(text)
        self.allow.setEnabled(True)
        self.keyboard.setEnabled(True)
        self.skip.setEnabled(True)

    def reject(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            return
        super().reject()

    def closeEvent(self, event):
        if self.process and self.process.state() != QProcess.NotRunning:
            event.ignore()
        else:
            super().closeEvent(event)
