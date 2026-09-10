import math
import pytest
from airmouse.config import Settings
from airmouse import gestures
from airmouse.gestures import Features, GestureMachine, State
from airmouse.mapping import CursorMapper, AdaptiveEMA
from airmouse.controller import Controller

S = Settings()
NEUTRAL = Features((.5,.5), .8,.8,False)
PINCH = Features((.5,.5), .1,.8,False)
RIGHT = Features((.5,.5), .8,.1,False)

def armed():
    m = GestureMachine()
    m.step(NEUTRAL,0,S,True)
    m.step(NEUTRAL,.4,S,True)
    return m

def test_paused_never_emits_press_or_move():
    m = GestureMachine()
    for t in range(10):
        assert m.step(PINCH,t,S,False) == []
        assert m.state == State.PAUSED

def test_entering_with_pinch_must_open_before_arming():
    m = GestureMachine()
    for t in range(10): assert m.step(PINCH,t,S,True) == []
    assert not m.armed
    m.step(NEUTRAL,10,S,True)
    m.step(NEUTRAL,10.4,S,True)
    assert m.armed

def test_click_drag_release_once():
    m = armed()
    assert m.step(PINCH,1,S,True) == []
    assert m.step(PINCH,1.1,S,True) == [('down',)]
    assert m.step(PINCH,1.4,S,True) == [('move', PINCH.point)]
    assert m.state == State.DRAGGING
    assert m.step(NEUTRAL,1.5,S,True) == [('up',)]
    assert m.step(NEUTRAL,1.6,S,True) == [('move',NEUTRAL.point)]

def test_jitter_hysteresis_and_short_pinch():
    m = armed()
    m.step(PINCH,1,S,True)
    assert m.step(NEUTRAL,1.02,S,True) == [('move', NEUTRAL.point)]
    m.step(PINCH,2,S,True)
    m.step(PINCH,2.1,S,True)
    mid = Features((.5,.5),.35,.8,False)
    assert m.step(mid,2.2,S,True)[0][0] == 'move'
    assert m.down

@pytest.mark.parametrize('paused',[True,False])
def test_tracking_loss_and_pause_release_drag(paused):
    m = armed()
    m.step(PINCH,1,S,True)
    m.step(PINCH,1.1,S,True)
    if paused:
        # An emergency stop is not a gesture: it releases on the spot, with no
        # grace period to ride out.
        assert m.step(None,1.2,S,False) == [('up',)]
    else:
        # A held drag latches through frames it cannot read, then gives up once
        # the hand has been unreadable for longer than the grace period.
        assert m.step(None,1.2,S,True) == []
        assert m.down
        assert m.step(None,1.1+S.drag_grace+.01,S,True) == [('up',)]
        assert m.release_reason == gestures.RELEASE_TIMEOUT
    assert not m.armed
    assert m.step(None,1.5,S,not paused) == []

def test_right_click_is_latched_until_open():
    m = armed()
    m.step(RIGHT,1,S,True)
    assert m.step(RIGHT,1.1,S,True) == [('right',)]
    for t in (2,3,4): assert ('right',) not in m.step(RIGHT,t,S,True)
    m.step(NEUTRAL,5,S,True)
    m.step(RIGHT,6,S,True)
    assert m.step(RIGHT,6.1,S,True) == [('right',)]

def test_scroll_engage_delay_and_freezes_cursor():
    m = armed()
    f = Features((.5,.5),.8,.8,True)
    assert m.step(f,1,S,True) == []                 # engaging
    assert m.step(f,1.1,S,True) == []               # .1s < .15s engage delay: frozen
    assert m.step(f,1.2,S,True) == []               # engaged, but no vertical motion yet
    scrolled = m.step(Features((.5,.4),.8,.8,True),1.3,S,True)   # ~.1 up * gain 50
    assert scrolled[0][0] == 'scroll' and scrolled[0][1] >= 4
    assert m.state == State.SCROLLING

def test_fast_scroll_does_not_drop_capped_steps():
    m = armed()
    fast = Features((.5,.1),.8,.8,True)             # large upward displacement
    m.step(Features((.5,.5),.8,.8,True),1,S,True)   # engage start
    m.step(Features((.5,.5),.8,.8,True),1.2,S,True) # engaged, anchor at .5
    # (.5-.1)*50 = 20 steps requested, capped to 6; the remainder must carry
    # into the next frame instead of being lost, so scrolling continues.
    assert m.step(fast,1.3,S,True) == [('scroll',6)]
    assert m.step(fast,1.35,S,True) == [('scroll',6)]

def test_gestures_can_be_disabled():
    s = Settings(left=False,right=False,scroll=False)
    m = armed()
    for f in (PINCH, RIGHT, Features((.5,.2),.8,.8,True)):
        assert m.step(f,1,s,True) == [('move',f.point)]

def test_mapping_edges_negative_desktop_and_clamps():
    mapper = CursorMapper((-1920,0,3840,1080))
    assert mapper.target((S.margin,S.margin),S) == (-1920,0)
    assert mapper.target((1-S.margin,1-S.margin),S) == (1919,1079)
    assert mapper.target((-1,2),S) == (-1920,1079)

def test_filter_jitter_and_convergence():
    f = AdaptiveEMA()
    f.reset((500,500))
    assert f.update((501,500),.03,.09,2.5) == (500,500)
    first = f.update((1000,500),.03,.09,2.5)
    assert 500 < first[0] < 1000
    for _ in range(100): last = f.update((1000,500),.03,.09,2.5)
    assert abs(last[0]-1000)<3

class FakeInput:
    def __init__(self): self.events=[]
    def move(self,*p): self.events.append(('move',p))
    def down(self): self.events.append(('down',))
    def up(self): self.events.append(('up',))
    def right(self): self.events.append(('right',))
    def scroll(self,n): self.events.append(('scroll',n))
    def close(self): pass

def test_controller_emergency_releases_and_blocks_late_frames():
    backend = FakeInput()
    c = Controller(S,(0,0,1920,1080),backend)
    c.resume()
    for t in (0,.1,.2,.4): c.process(NEUTRAL,t)
    c.process(PINCH,.5)
    c.process(PINCH,.6)
    assert ('down',) in backend.events
    c.pause()
    count = len(backend.events)
    for t in (.7,.8,1): c.process(PINCH,t)
    assert len(backend.events) == count
    assert backend.events[-1] == ('up',)

def test_initial_movement_is_speed_limited():
    backend = FakeInput()
    c = Controller(S,(0,0,1920,1080),backend)
    c.resume()
    edge = Features((1,1),.8,.8,False)
    for t in (0,.1,.2,.4,.43): c.process(edge,t)
    point = backend.events[-1][1]
    assert math.dist((960,540),point) < 110

def test_bad_calibration_rejected():
    with pytest.raises(ValueError): Settings(pinch=.5,release=.2).validate()
    with pytest.raises(ValueError): Settings(smoothing=float('nan')).validate()
    with pytest.raises(ValueError): Settings(precision_gain=1.5).validate()
    with pytest.raises(ValueError): Settings(precision_gain=0).validate()
    with pytest.raises(ValueError): Settings(drag_grace=.9).validate()
    with pytest.raises(ValueError): Settings(precision_assist='yes').validate()

def test_a_settings_file_without_the_new_fields_keeps_its_calibration(monkeypatch,tmp_path):
    import json
    from dataclasses import asdict
    monkeypatch.setattr('airmouse.config.CONFIG_PATH',tmp_path/'settings.json')
    fresh = {k:v for k,v in asdict(Settings(sensitivity=1.5,smoothing=.2)).items()
             if k not in ('precision_assist','precision_gain','drag_grace')}
    (tmp_path/'settings.json').write_text(json.dumps(fresh))
    loaded = Settings.load()
    assert (loaded.sensitivity,loaded.smoothing) == (1.5,.2)
    assert loaded.drag_grace == Settings().drag_grace

def test_an_unknown_setting_does_not_discard_the_rest_of_the_file(monkeypatch,tmp_path):
    """A file from a newer build used to fail the whole constructor and reset
    the user's entire calibration over one key this build had never heard of."""
    import json
    from dataclasses import asdict
    monkeypatch.setattr('airmouse.config.CONFIG_PATH',tmp_path/'settings.json')
    stored = asdict(Settings(sensitivity=1.75))
    stored['setting_from_a_later_build'] = 42
    (tmp_path/'settings.json').write_text(json.dumps(stored))
    assert Settings.load().sensitivity == 1.75

def test_the_retired_velocity_gain_settings_are_carried_across(monkeypatch,tmp_path):
    """The velocity-gain curve is gone -- it made the pointer history-dependent --
    but the switch and the slow-end gain still mean something, so a settings
    file written against the old names must not lose them. `gain_max` described
    high-speed lead, which no longer exists to configure."""
    import json
    from dataclasses import asdict
    monkeypatch.setattr('airmouse.config.CONFIG_PATH',tmp_path/'settings.json')
    stored = {k:v for k,v in asdict(Settings()).items()
              if k not in ('precision_assist','precision_gain')}
    stored.update(pointer_gain=False, gain_min=.3, gain_max=1.9)
    (tmp_path/'settings.json').write_text(json.dumps(stored))
    loaded = Settings.load()
    assert loaded.precision_assist is False
    assert loaded.precision_gain == .3

def test_a_corrupt_settings_file_still_falls_back_to_defaults(monkeypatch,tmp_path):
    monkeypatch.setattr('airmouse.config.CONFIG_PATH',tmp_path/'settings.json')
    for content in ('not json at all','[]','{"sensitivity": 99}'):
        (tmp_path/'settings.json').write_text(content)
        assert Settings.load().sensitivity == Settings().sensitivity

def test_stalled_processing_releases_drag_and_requires_reacquisition():
    backend = FakeInput()
    c = Controller(S,(0,0,1920,1080),backend)
    c.resume()
    for t in (0,.1,.2,.4): c.process(NEUTRAL,t)
    c.process(PINCH,.5)
    c.process(PINCH,.6)
    assert c.machine.down
    c.process(PINCH,2)
    assert backend.events[-1] == ('up',)
    assert not c.machine.armed
    c.process(PINCH,2.1)
    assert not c.machine.down

def test_cooldown_prevents_immediate_second_press():
    m = armed()
    m.step(PINCH,1,S,True)
    m.step(PINCH,1.1,S,True)
    m.step(NEUTRAL,1.2,S,True)
    m.step(PINCH,1.21,S,True)
    assert m.step(PINCH,1.31,S,True) == []
    assert not m.down
    assert m.step(PINCH,1.6,S,True) == [('down',)]

def test_lighter_pinch_registers_a_click():
    # A pinch at .30 sits above the old .28 threshold but below the current .32,
    # so a lighter, more natural pinch now clicks.
    m = armed()
    light = Features((.5,.5), .30, .8, False)
    m.step(light,1,S,True)
    assert m.step(light,1.1,S,True) == [('down',)]

def test_pinch_confirms_within_shortened_debounce():
    m = armed()
    m.step(PINCH,1,S,True)
    # Held only 60 ms: below the old 80 ms debounce, at/above the current 50 ms.
    assert m.step(PINCH,1.06,S,True) == [('down',)]

def test_cooldown_allows_faster_repeat_click():
    m = armed()
    m.step(PINCH,1,S,True)
    m.step(PINCH,1.1,S,True)          # first press
    m.step(NEUTRAL,1.2,S,True)        # release, last action at 1.2
    m.step(PINCH,1.25,S,True)
    # 300 ms after the release: blocked by the old .35 cooldown, allowed by .25.
    assert m.step(PINCH,1.5,S,True) == [('down',)]

def scroll_pose_points(fold=True):
    """21 landmarks with index+middle raised; ring+little folded (fold=True) or
    raised into an open palm (fold=False)."""
    pts = {
        0:(.50,.90), 1:(.42,.78), 2:(.38,.72), 3:(.36,.66), 4:(.36,.60),
        5:(.45,.62), 6:(.45,.50), 7:(.45,.40), 8:(.45,.32),
        9:(.50,.60), 10:(.50,.47), 11:(.50,.37), 12:(.50,.30)}
    if fold:
        pts.update({13:(.55,.62),14:(.56,.52),15:(.55,.58),16:(.54,.63),
                    17:(.60,.66),18:(.61,.58),19:(.60,.62),20:(.59,.66)})
    else:
        pts.update({13:(.55,.62),14:(.56,.50),15:(.57,.40),16:(.58,.32),
                    17:(.62,.64),18:(.64,.54),19:(.66,.45),20:(.68,.38)})
    return [(pts[i][0],pts[i][1],0.) for i in range(21)]

def test_scroll_pose_detected_from_landmarks():
    assert Features.from_landmarks(scroll_pose_points(fold=True), 1.).scroll

def test_open_palm_is_not_a_scroll_pose():
    assert not Features.from_landmarks(scroll_pose_points(fold=False), 1.).scroll


def test_camera_take_blocks_until_a_frame_lands():
    """The worker sleeps on the capture thread rather than polling, so a frame
    is picked up as soon as it exists instead of up to a poll interval later."""
    import threading, time
    from airmouse.capture import Camera
    camera = Camera.__new__(Camera)          # no webcam needed for the handoff
    camera.lock = threading.Condition()
    camera.latest = None
    assert camera.take(0.02) is None         # nothing published yet

    def publish():
        time.sleep(0.03)
        with camera.lock:
            camera.latest = (1.0, 'frame')
            camera.lock.notify()
    threading.Thread(target=publish, daemon=True).start()
    start = time.monotonic()
    item = camera.take(2.0)
    waited = time.monotonic() - start
    assert item == (1.0, 'frame')
    assert waited < 0.5                      # woken by the notify, not the timeout
    assert camera.take(0) is None            # slot cleared after the take
