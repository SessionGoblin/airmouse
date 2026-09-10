import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import math
import time

import pytest
from PySide6.QtWidgets import QApplication

from airmouse import poses, recorder
from airmouse.config import Settings
from airmouse.gestures import Features
from airmouse.roles import Hand, POINTER, MODIFIER

app = QApplication.instance() or QApplication([])


def hand(seed=1, jitter=0.0):
    import random
    rng = random.Random(seed)
    points = [(rng.uniform(.3, .7) + rng.uniform(-jitter, jitter),
               rng.uniform(.2, .8) + rng.uniform(-jitter, jitter), 0.0) for _ in range(21)]
    points[poses.WRIST] = (.5, .8, 0.0)
    points[poses.MIDDLE_MCP] = (.5, .5, 0.0)
    return points


def features_for(points, left=.8, right=.8, scroll=False):
    pose, orientation, chirality = poses.normalize(points, 1.0)
    return Features((.5, .5), left, right, scroll, pose, orientation, chirality)


def as_hands(features, role=POINTER):
    return [] if features is None else [Hand(points=[(0., 0., 0.)]*21, role=role,
                                             features=features)]


class Feed:
    """Stands in for the window's per-frame publication of tracked hands."""
    def __init__(self, features, role=POINTER):
        self.features = features
        self.role = role
        self.stamp = time.monotonic()

    def __call__(self):
        return as_hands(self.features, self.role), self.stamp


def drive(dialog, seconds, step=.02):
    """Advance the recorder's clock without waiting in real time."""
    dialog.started = time.monotonic() - seconds
    for _ in range(int(seconds/step) + 4):
        dialog.sample()


def test_recording_averages_steady_frames_into_a_template():
    feed = Feed(features_for(hand()))
    dialog = recorder.RecordDialog(feed, 'wave')
    dialog.timer.stop()
    drive(dialog, recorder.COUNTDOWN + recorder.SAMPLE_SECONDS + .1)
    assert dialog.template is not None
    assert dialog.template.name == 'wave'
    assert len(dialog.template.pose) == poses.LANDMARKS
    # The template matches the pose it was recorded from.
    assert dialog.template.matches(feed.features.pose, feed.features.orientation) is not None


def test_countdown_runs_before_any_frame_is_kept():
    feed = Feed(features_for(hand()))
    dialog = recorder.RecordDialog(feed, 'wave')
    dialog.timer.stop()
    drive(dialog, recorder.COUNTDOWN - .5)
    assert dialog.samples == []
    assert 'Hold the pose' in dialog.message.text()


def test_a_hand_that_disappears_restarts_rather_than_averaging_the_gap():
    feed = Feed(features_for(hand()))
    dialog = recorder.RecordDialog(feed, 'wave')
    dialog.timer.stop()
    drive(dialog, recorder.COUNTDOWN + .4)
    assert dialog.samples
    feed.features = None
    dialog.sample()
    assert dialog.samples == []
    assert 'No pointer hand' in dialog.message.text()


def test_stale_features_count_as_no_hand():
    """A stopped camera leaves the last features in place; only the timestamp
    reveals that they are no longer live."""
    feed = Feed(features_for(hand()))
    feed.stamp = time.monotonic() - (recorder.FRESH + .2)
    dialog = recorder.RecordDialog(feed, 'wave')
    dialog.timer.stop()
    drive(dialog, recorder.COUNTDOWN + 1)
    assert dialog.samples == []
    assert dialog.template is None


def test_an_unsteady_pose_is_refused_rather_than_saved():
    """Averaging a hand that never settled produces a template that matches
    nothing, so it is better to fail loudly at record time."""
    class Shaky:
        def __init__(self):
            self.n = 0
            self.stamp = time.monotonic()

        def __call__(self):
            self.n += 1
            self.stamp = time.monotonic()
            return as_hands(features_for(hand(seed=self.n))), self.stamp

    dialog = recorder.RecordDialog(Shaky(), 'jitter')
    dialog.timer.stop()
    drive(dialog, recorder.COUNTDOWN + recorder.SAMPLE_SECONDS + .1)
    assert dialog.template is None
    assert 'unsteady' in dialog.message.text()


def test_shadow_check_reports_built_ins_the_pose_would_steal():
    """The built-ins are distance thresholds, not poses, so the overlap is
    measured from the recorded pinch ratios."""
    settings = Settings()
    feed = Feed(features_for(hand(), left=.05))          # a fist-like pinch
    dialog = recorder.RecordDialog(feed, 'fist')
    dialog.timer.stop()
    drive(dialog, recorder.COUNTDOWN + recorder.SAMPLE_SECONDS + .1)
    assert dialog.shadows(settings) == ['pinch click']
    # A neutral hand shadows nothing.
    open_feed = Feed(features_for(hand()))
    other = recorder.RecordDialog(open_feed, 'open')
    other.timer.stop()
    drive(other, recorder.COUNTDOWN + recorder.SAMPLE_SECONDS + .1)
    assert other.shadows(settings) == []


def test_shadow_check_respects_disabled_built_ins():
    feed = Feed(features_for(hand(), left=.05))
    dialog = recorder.RecordDialog(feed, 'fist')
    dialog.timer.stop()
    drive(dialog, recorder.COUNTDOWN + recorder.SAMPLE_SECONDS + .1)
    assert dialog.shadows(Settings(left=False)) == []


# --- library dialog ------------------------------------------------------------

def gesture_dialog(templates, tmp_path, monkeypatch):
    monkeypatch.setattr('airmouse.poses.GESTURE_PATH', tmp_path/'gestures.json')
    return recorder.GestureDialog(poses.Library(list(templates)), Settings(),
                                  lambda: ([], None))


def make_template(name, seed=1, **kw):
    pose, orientation, _ = poses.normalize(hand(seed), 1.0)
    return poses.Template(name=name, pose=pose, threshold=.2, **kw)


def test_save_rejects_an_invalid_shortcut_and_keeps_the_dialog_open(tmp_path, monkeypatch):
    d = gesture_dialog([make_template('a', action='key', argument='ctrl+nope')],
                       tmp_path, monkeypatch)
    d.apply()
    from PySide6.QtWidgets import QDialog
    assert d.result() != QDialog.Accepted
    assert 'Unknown key' in d.error.text()
    assert not (tmp_path/'gestures.json').exists()


def test_save_writes_normalized_bindings(tmp_path, monkeypatch):
    d = gesture_dialog([make_template('a', action='key', argument='CTRL+ALT+T')],
                       tmp_path, monkeypatch)
    d.apply()
    saved = poses.Library.load(tmp_path/'gestures.json')
    assert saved.templates[0].argument == 'ctrl+alt+t'


def test_editing_the_binding_writes_back_to_the_template(tmp_path, monkeypatch):
    d = gesture_dialog([make_template('a')], tmp_path, monkeypatch)
    d.kind.setCurrentIndex(d.kind.findData('shell'))
    d.argument.setText('firefox')
    d.store()
    assert d.current().action == 'shell'
    assert d.current().argument == 'firefox'


def test_tilt_checkbox_switches_orientation_matching(tmp_path, monkeypatch):
    """Thumbs-up and thumbs-down are the same shape; the checkbox is what makes
    them separable."""
    d = gesture_dialog([make_template('a')], tmp_path, monkeypatch)
    assert d.current().orientation is None      # tilt-invariant by default
    d.tilt.setChecked(True)
    assert d.current().orientation is not None
    d.tilt.setChecked(False)
    assert d.current().orientation is None


def test_recorded_orientation_is_not_persisted(tmp_path, monkeypatch):
    d = gesture_dialog([make_template('a', action='shell', argument='true')],
                       tmp_path, monkeypatch)
    d.current().recorded_orientation = 1.23
    d.apply()
    raw = (tmp_path/'gestures.json').read_text()
    assert 'recorded_orientation' not in raw
    assert poses.Library.load(tmp_path/'gestures.json').templates[0].name == 'a'


def test_delete_removes_the_template(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    d = gesture_dialog([make_template('a'), make_template('b', seed=5)], tmp_path, monkeypatch)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.Yes)
    d.list.setCurrentRow(0)
    d.remove()
    assert [t.name for t in d.library.templates] == ['b']


def test_dialog_edits_a_copy_until_saved(tmp_path, monkeypatch):
    """Cancelling must not mutate the library the controller is matching
    against on the vision thread."""
    original = poses.Library([make_template('a', action='shell', argument='true')])
    monkeypatch.setattr('airmouse.poses.GESTURE_PATH', tmp_path/'gestures.json')
    d = recorder.GestureDialog(original, Settings(), lambda: ([], None))
    d.library.remove('a')
    assert [t.name for t in original.templates] == ['a']


def test_record_buttons_survive_the_clicked_signal(tmp_path, monkeypatch):
    """clicked emits `checked`, and PySide passes it to any slot that can take
    an argument -- so record(existing=None) received a bool. Only pressing the
    real button catches this; calling record() directly does not."""
    seen = []

    d = gesture_dialog([make_template('a')], tmp_path, monkeypatch)
    monkeypatch.setattr(d, 'record', lambda existing=None: seen.append(existing))
    # Reconnect through the same lambdas the dialog builds.
    d.record_button.click()
    d.list.setCurrentRow(0)
    d.rerecord.click()
    assert seen[0] is None                      # new pose, not False
    assert seen[1] is d.library.templates[0]    # re-record targets the selection


def test_record_button_creates_a_uniquely_named_template(tmp_path, monkeypatch):
    """Capture is non-modal now, so the result arrives through `finished`
    rather than from exec(); drive that path directly."""
    built = make_template('Gesture 1', seed=4)

    class StubRecord(recorder.RecordDialog):
        def __init__(self, source, name, parent=None, role='pointer'):
            super().__init__(source, name, parent, role=role)
            self.timer.stop()
            self.template = poses.Template(name=name, pose=built.pose,
                                           threshold=.2, role=role)

        def shadows(self, settings):
            return []

    monkeypatch.setattr(recorder, 'RecordDialog', StubRecord)
    d = gesture_dialog([make_template('a')], tmp_path, monkeypatch)
    d.record_button.click()
    d.capture.accept()                  # the user finishes the recording
    assert [t.name for t in d.library.templates] == ['a', 'Gesture 1']
    assert d.capture is None and d.isEnabled()
    d.record_button.click()
    d.capture.accept()
    assert [t.name for t in d.library.templates] == ['a', 'Gesture 1', 'Gesture 2']


def test_the_list_is_disabled_while_a_recording_is_in_flight(tmp_path, monkeypatch):
    """Non-modal capture must still stop the list being edited underneath it."""
    class StubRecord(recorder.RecordDialog):
        def __init__(self, source, name, parent=None, role='pointer'):
            super().__init__(source, name, parent, role=role)
            self.timer.stop()

    monkeypatch.setattr(recorder, 'RecordDialog', StubRecord)
    d = gesture_dialog([make_template('a')], tmp_path, monkeypatch)
    d.record_button.click()
    assert d.capture is not None and not d.isEnabled()
    d.record_button.click()             # a second press must not stack dialogs
    assert d.capture is not None
    d.capture.reject()                  # cancelled
    assert d.capture is None and d.isEnabled()
    assert [t.name for t in d.library.templates] == ['a']


# --- capture dialog layout -----------------------------------------------------

RECORD_MESSAGES = ['Get ready…', 'Hold the pose… 3', 'Recording…',
                   'No pointer hand visible', 'No modifier hand visible',
                   'Pose was too unsteady']
DRAW_MESSAGES = ['Get ready…', 'Pinch the modifier hand to start', 'Drawing…',
                 'No camera frames', 'Show your modifier hand', 'Show your pointer hand',
                 'Stroke not usable']
DETAILS = ['only 3 points captured — hold the gate steady through the whole stroke',
           'the hand barely moved — draw the shape larger', '42 points',
           'Only 4 usable frames. Close and try again, holding the pose still under '
           'even lighting.']


def capture_dialogs():
    record = recorder.RecordDialog(lambda: ([], None), 'test', role='modifier')
    record.timer.stop()
    draw = recorder.DrawDialog(lambda: ([], None), 'swipe')
    draw.timer.stop()
    return [(record, RECORD_MESSAGES), (draw, DRAW_MESSAGES)]


def test_no_capture_message_is_clipped():
    """The message label started empty, so the dialog sized itself around a
    blank 20px label and cut the glyphs off the first real message."""
    for dialog, messages in capture_dialogs():
        dialog.show()
        for text in messages:
            dialog.message.setText(text)
            app.processEvents()
            assert dialog.message.sizeHint().height() <= dialog.message.height(), \
                f'clipped: {text!r}'
        dialog.close()


def test_the_capture_dialog_does_not_resize_as_its_message_changes():
    """These change on a 25 ms timer; a dialog that refits each one jitters."""
    for dialog, messages in capture_dialogs():
        dialog.show()
        sizes = set()
        for text in messages:
            dialog.message.setText(text)
            app.processEvents()
            sizes.add((dialog.width(), dialog.height()))
        assert len(sizes) == 1, f'geometry moved across messages: {sizes}'
        dialog.close()


def test_detail_text_fits_without_clipping():
    for dialog, _ in capture_dialogs():
        dialog.show()
        for text in DETAILS:
            dialog.detail.setText(text)
            app.processEvents()
            assert dialog.detail.sizeHint().height() <= dialog.detail.height(), \
                f'clipped: {text!r}'
        dialog.close()


def test_the_message_starts_with_real_text():
    """An empty label is what let the dialog size itself wrongly."""
    for dialog, _ in capture_dialogs():
        assert dialog.message.text().strip()
        dialog.close()
