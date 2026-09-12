import os
import sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import stat
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication, QDialog
from airmouse import permission_helper as helper
from airmouse import permissions

app = QApplication.instance() or QApplication([])


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux permission helper')
def test_helper_limits_grants_to_authenticated_user_and_selected_keyboard(monkeypatch):
    calls = []
    monkeypatch.setenv('PKEXEC_UID', '1000')
    monkeypatch.setattr(helper.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(helper, 'keyboards', lambda: [('/dev/input/event5', 'Keyboard')])
    monkeypatch.setattr(helper.os, 'lstat', lambda path: SimpleNamespace(st_mode=stat.S_IFCHR))
    monkeypatch.setattr(helper.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    helper.grant('/dev/input/event5')
    assert calls == [
        ['/usr/sbin/modprobe', 'uinput'],
        ['/usr/bin/setfacl', '-m', 'u:1000:rw', '/dev/uinput'],
        ['/usr/bin/setfacl', '-m', 'u:1000:r', '/dev/input/event5'],
    ]
    calls.clear()
    for invalid in ['/etc/passwd', '/dev/input/event6', '/dev/input/event5; id']:
        with pytest.raises(RuntimeError):
            helper.grant(invalid)
    assert not calls


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux permission helper')
def test_helper_refuses_without_pkexec_caller(monkeypatch):
    monkeypatch.delenv('PKEXEC_UID', raising=False)
    with pytest.raises(RuntimeError):
        helper.grant('/dev/input/event5')


def dialog(monkeypatch):
    monkeypatch.setattr(permissions, 'keyboards', lambda: [('/dev/input/event5', 'Keyboard')])
    monkeypatch.setattr(permissions.os, 'access', lambda *args: True)
    return permissions.PermissionDialog()


def test_dialog_skip_never_starts_privileged_process(monkeypatch):
    d = dialog(monkeypatch)
    assert d.process is None
    d.skip.click()
    assert d.result() == QDialog.Rejected
    assert d.process is None


def test_authentication_failure_allows_retry_and_preview(monkeypatch):
    d = dialog(monkeypatch)
    d.process = SimpleNamespace(readAllStandardError=lambda: b'Authentication cancelled')
    d.finished_grant(126, QProcess.NormalExit)
    assert d.result() == QDialog.Rejected
    assert d.allow.isEnabled() and d.skip.isEnabled()
    assert 'Authentication cancelled' in d.message.text()
    d.process = None
    d.close()


def test_success_requires_verified_access(monkeypatch):
    d = dialog(monkeypatch)
    d.process = SimpleNamespace(readAllStandardError=lambda: b'')
    monkeypatch.setattr(permissions, 'access_ready', lambda: False)
    d.finished_grant(0, QProcess.NormalExit)
    assert d.result() == QDialog.Rejected
    monkeypatch.setattr(permissions, 'access_ready', lambda: True)
    d.finished_grant(0, QProcess.NormalExit)
    assert d.result() == QDialog.Accepted
    d.process = None
    d.close()
