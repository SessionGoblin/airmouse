"""Model startup must also work from the Windows windowed launcher."""
from airmouse import tracking


def test_native_stderr_context_without_console(monkeypatch):
    monkeypatch.setattr(tracking.sys, 'stderr', None)
    with tracking._muted_native_stderr():
        reached = True
    assert reached


def test_native_stderr_context_without_descriptor(monkeypatch):
    def unavailable(fd):
        raise OSError('Bad file descriptor')
    monkeypatch.setattr(tracking.os, 'dup', unavailable)
    with tracking._muted_native_stderr():
        reached = True
    assert reached
