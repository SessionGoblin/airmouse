import pytest

from airmouse import actions, poses
from airmouse.config import Settings
from airmouse.controller import Controller

BOUNDS = (0, 0, 1920, 1080)


class FakeKeyboard:
    def __init__(self):
        self.taps = []
        self.closed = False

    def tap(self, tokens):
        self.taps.append(tuple(tokens))

    def close(self):
        self.closed = True


def test_shortcut_parsing_normalizes_and_orders():
    assert actions.parse_keys('Ctrl+Alt+T') == ('ctrl', 'alt', 't')
    assert actions.parse_keys(' super ') == ('super',)
    assert actions.validate('key', 'CTRL+ALT+t') == 'ctrl+alt+t'


@pytest.mark.parametrize('spec, message', [
    ('', 'Empty'),
    ('ctrl+nope', 'Unknown key'),
    ('ctrl+shift', 'modifier'),
    ('ctrl+ctrl+t', 'Repeated'),
])
def test_bad_shortcuts_are_rejected_at_entry(spec, message):
    with pytest.raises(ValueError, match=message):
        actions.parse_keys(spec)


def test_the_apps_own_hotkeys_cannot_be_bound():
    """Injecting F8 or F12 would be read straight back by the hotkey listener,
    so a pose could pause the app or fight it every frame."""
    for reserved in ('f8', 'f12', 'ctrl+F12'):
        with pytest.raises(ValueError, match='reserved'):
            actions.parse_keys(reserved)
    assert actions.parse_keys('f9') == ('f9',)


def test_app_and_shell_bindings_are_validated():
    assert actions.validate('app', 'toggle') == 'toggle'
    with pytest.raises(ValueError, match='Unknown app action'):
        actions.validate('app', 'explode')
    assert actions.validate('shell', '  firefox --new-window ') == 'firefox --new-window'
    with pytest.raises(ValueError, match='Empty command'):
        actions.validate('shell', '   ')
    with pytest.raises(ValueError, match='Unknown action'):
        actions.validate('teleport', 'x')


def primed(factory):
    """A dispatcher whose keyboard has finished building."""
    d = actions.Dispatcher(keyboard_factory=factory)
    d.prime()
    d._priming.join(2)
    return d


def test_key_binding_taps_and_reuses_one_device():
    keyboard = FakeKeyboard()
    made = []
    d = primed(lambda: made.append(1) or keyboard)
    assert d.run(poses.Template(name='a', action='key', argument='ctrl+alt+t'))
    assert d.run(poses.Template(name='b', action='key', argument='super'))
    assert keyboard.taps == [('ctrl', 'alt', 't'), ('super',)]
    assert made == [1]                      # built once, not per press


def test_a_key_press_never_waits_for_the_device_to_be_built():
    """Building it takes up to two seconds while evdev waits on udev. Inline,
    on the vision thread, that stalls tracking past the watchdog and control is
    disabled -- which happened on the first key gesture and never again."""
    import threading, time
    release = threading.Event()
    keyboard = FakeKeyboard()

    def slow():
        release.wait(5)
        return keyboard

    d = actions.Dispatcher(keyboard_factory=slow)
    d.prime()
    start = time.monotonic()
    fired = d.run(poses.Template(name='a', action='key', argument='ctrl+w'))
    assert time.monotonic() - start < .2       # returned immediately
    assert fired is False                      # this press is dropped, not blocked
    assert 'still starting up' in d.error
    release.set()
    d._priming.join(2)
    assert d.ready()
    assert d.run(poses.Template(name='a', action='key', argument='ctrl+w'))
    assert keyboard.taps == [('ctrl', 'w')]


def test_an_unprimed_key_press_starts_the_build_itself():
    """Priming is a head start, not a requirement: a binding added mid-session
    still works, it just loses the first press."""
    keyboard = FakeKeyboard()
    d = actions.Dispatcher(keyboard_factory=lambda: keyboard)
    assert d.run(poses.Template(name='a', action='key', argument='ctrl+w')) is False
    d._priming.join(2)
    assert d.run(poses.Template(name='a', action='key', argument='ctrl+w'))


def test_priming_twice_builds_one_device():
    made = []
    d = actions.Dispatcher(keyboard_factory=lambda: made.append(1) or FakeKeyboard())
    d.prime()
    d.prime()
    if d._priming:
        d._priming.join(2)
    d.prime()
    assert made == [1]


def test_a_keyboard_that_cannot_be_built_reports_instead_of_raising():
    def broken():
        raise OSError('no /dev/uinput')
    d = actions.Dispatcher(keyboard_factory=broken)
    d.prime()
    d._priming.join(2)
    assert not d.ready()
    assert 'keyboard unavailable' in d.error
    assert d.run(poses.Template(name='a', action='key', argument='ctrl+w')) is False


def test_no_keyboard_device_until_a_key_binding_fires():
    """A setup that only uses app and shell bindings should never ask for the
    extra uinput node."""
    made = []
    d = actions.Dispatcher(app=lambda a: None, spawn=lambda c: None,
                           keyboard_factory=lambda: made.append(1))
    d.run(poses.Template(name='a', action='app', argument='toggle'))
    d.run(poses.Template(name='b', action='shell', argument='true'))
    assert made == []


def test_shell_binding_spawns_the_parsed_command():
    ran = []
    d = actions.Dispatcher(spawn=ran.append)
    assert d.run(poses.Template(name='browser', action='shell', argument='firefox --safe'))
    assert ran == ['firefox --safe']


def test_app_binding_reaches_the_callback():
    seen = []
    d = actions.Dispatcher(app=seen.append)
    assert d.run(poses.Template(name='p', action='app', argument='pause'))
    assert seen == ['pause']


def test_unbound_template_does_nothing():
    d = actions.Dispatcher()
    assert d.run(poses.Template(name='idle')) is False
    assert d.error == ''


def test_a_failing_binding_records_an_error_instead_of_raising():
    """The dispatcher runs on the vision thread; an exception there would stop
    tracking entirely."""
    def boom(command):
        raise FileNotFoundError('no such program')
    d = actions.Dispatcher(spawn=boom)
    assert d.run(poses.Template(name='gone', action='shell', argument='nope')) is False
    assert 'gone: no such program' in d.error
    # A later good binding clears it.
    d._spawn = lambda c: None
    assert d.run(poses.Template(name='ok', action='shell', argument='true'))
    assert d.error == ''


def test_spawn_passes_argv_with_no_shell(monkeypatch):
    """The command lives in a config file and fires on a gesture; it must not
    quietly gain glob expansion or operator chaining."""
    calls = {}

    def fake_popen(argv, **kwargs):
        calls['argv'], calls['kwargs'] = argv, kwargs
    monkeypatch.setattr(actions.subprocess, 'Popen', fake_popen)
    actions._spawn('notify-send "hello world" && rm -rf ~')
    # Metacharacters arrive as literal arguments, not as shell syntax.
    assert calls['argv'] == ['notify-send', 'hello world', '&&', 'rm', '-rf', '~']
    assert not calls['kwargs'].get('shell')
    assert calls['kwargs']['start_new_session'] is True


# --- controller integration ----------------------------------------------------

def armed_controller(template, dispatcher):
    library = poses.Library([template])
    controller = Controller(Settings(), BOUNDS, backend=None, library=library,
                            dispatcher=dispatcher)
    controller.enabled = True
    controller.machine.armed = True
    return controller


def hold(controller, pose, start=1.0, fps=30):
    """Hold a pose across real frame intervals until its dwell clears.

    Stepping straight to the dwell in one jump would trip the lost-hand reset,
    which fires on any gap over .3 s.
    """
    from airmouse.gestures import Features
    f = Features((.5, .5), .8, .8, False, pose, 0.0)
    from airmouse.gestures import State
    state = None
    for i in range(int((Settings().custom_dwell + .1) * fps)):
        state = controller.process(f, start + i/fps)
        if state == State.CUSTOM.value:     # stop on the frame it fires
            break
    return state


def test_matched_pose_runs_its_binding_through_the_controller():
    pose = poses.normalize([(.5, .8, 0.)] + [(.4+i*.01, .3+i*.01, 0.) for i in range(20)], 1.0)[0]
    ran = []
    template = poses.Template(name='launch', action='shell', argument='xterm',
                              pose=pose, threshold=.5)
    controller = armed_controller(template, actions.Dispatcher(spawn=ran.append))
    hold(controller, pose)
    assert ran == ['xterm']


def test_recenter_clears_the_filter_rather_than_teleporting():
    """recenter is pointer state, so it re-enters the slow reacquisition path
    instead of jumping the cursor."""
    pose = poses.normalize([(.5, .8, 0.)] + [(.4+i*.01, .3+i*.01, 0.) for i in range(20)], 1.0)[0]
    template = poses.Template(name='home', action='app', argument='recenter',
                              pose=pose, threshold=.5)
    seen = []
    controller = armed_controller(template, actions.Dispatcher(app=seen.append))
    controller.mapper.filter.value = (900, 500)
    controller.target = (900, 500)
    hold(controller, pose)
    assert controller.mapper.filter.value is None
    assert controller.target is None
    assert seen == []                       # handled by the controller, not the dispatcher
