import math

from airmouse.config import Settings
from airmouse.controller import Controller
from airmouse.gestures import Features, GestureMachine, State
from airmouse.mapping import CursorMapper
from test_behavior import FakeInput, armed, NEUTRAL, RIGHT


def pointing_controller():
    backend = FakeInput()
    controller = Controller(Settings(), (0,0,1920,1080), backend)
    controller.resume()
    for t in (0,.1,.2,.4,.45):
        controller.process(NEUTRAL,t)
    return controller,backend


def test_pinch_closure_does_not_drag_click_target():
    c,b = pointing_controller()
    # Closing the pinch moves the fingertip left, but the click stays put.
    pinch = Features((.48,.5),.1,.8,False)
    c.process(pinch,.5)
    c.process(pinch,.6)
    count = len(b.events)
    c.process(Features((.481,.5),.1,.8,False),.65)
    assert len(b.events) == count
    assert b.events[-1] == ('down',)
    origin = c.target
    c.process(Features((.50,.5),.1,.8,False),.7)
    assert b.events[-1][0] == 'move'
    assert c.target[0] > origin[0]
    c.pause()
    assert b.events[-1] == ('up',)


def test_isolated_tracking_jump_cannot_click():
    c,b = pointing_controller()
    count = len(b.events)
    c.process(Features((.95,.1),.1,.8,False),.5)
    c.process(NEUTRAL,.55)
    assert not c.machine.armed
    assert len(b.events) == count


def test_nonfinite_tracking_releases_drag():
    """A single unreadable frame is now ridden out; sustained garbage still
    ends the drag, and still ends it as a release rather than as a throw."""
    c,b = pointing_controller()
    pinch = Features((.5,.5),.1,.8,False)
    c.process(pinch,.5)
    c.process(pinch,.6)
    bad = Features((math.nan,.5),.1,.8,False)
    c.process(bad,.65)
    assert c.machine.down                       # latched, not released
    assert b.events[-1] != ('up',)
    c.process(bad,.6+c.settings.drag_grace+.01)
    assert b.events[-1] == ('up',)
    assert not c.machine.armed
    assert c.throw is None


def test_slow_subpixel_motion_is_not_lost():
    mapper = CursorMapper((0,0,1920,1080))
    mapper.filter.reset((960,540))
    result = mapper.update((.501,.5), Settings(smoothing=.4,deadzone=0), .005)
    assert 960 < result[0] < 960.5
    assert result[0] != round(result[0])


def test_right_pinch_keeps_context_menu_target_stationary():
    m = armed()
    s = Settings()
    m.step(RIGHT,1,s,True)
    assert m.step(RIGHT,1.1,s,True) == [('right',)]
    assert m.step(Features((.7,.8),.8,.3,False),1.2,s,True) == []
    assert m.state == State.RIGHT_CLICK


def test_scroll_pose_flicker_does_not_move_pointer():
    m = armed()
    s = Settings()
    scroll = Features((.5,.5),.8,.8,True)
    m.step(scroll,1,s,True)
    m.step(scroll,1.4,s,True)
    assert m.step(NEUTRAL,1.45,s,True) == []
    assert m.step(scroll,1.5,s,True) == []
    assert m.state == State.SCROLLING
    assert m.step(NEUTRAL,1.6,s,True) == []
    assert m.step(NEUTRAL,1.73,s,True) == [('move',NEUTRAL.point)]


def test_reacquisition_keeps_previous_cursor_origin():
    c,b = pointing_controller()
    c.target = (1700.,850.)
    c.process(None,.5)
    c.process(NEUTRAL,.55)
    assert c.mapper.filter.value == (1700.,850.)
