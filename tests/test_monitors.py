import pytest

from airmouse.config import Settings
from airmouse.controller import Controller
from airmouse.gestures import Features
from airmouse.mapping import CursorMapper
from test_behavior import FakeInput


@pytest.mark.parametrize('screens, point, expected', [
    ([(0, 0, 100, 100), (200, 0, 100, 100)], (140, 50), (99, 50)),
    ([(0, 0, 100, 100), (200, 0, 100, 100)], (170, 50), (200, 50)),
    ([(0, -200, 100, 100), (0, 0, 100, 100)], (50, -40), (50, 0)),
    ([(-200, 0, 100, 100), (0, 50, 100, 100)], (-10, 10), (0, 50)),
    ([(0, 0, 100, 100), (100, 50, 100, 100)], (130, 80), (130, 80)),
    ([(0, 0, 100, 100), (0, 0, 100, 100)], (50.25, 80.5), (50.25, 80.5)),
])
def test_projects_onto_visible_monitor(screens, point, expected):
    mapper = CursorMapper((-200, -200, 500, 500), screens)
    assert mapper.visible_point(point) == expected


@pytest.mark.parametrize('drag', [False, True])
def test_pointer_crosses_large_gap_without_getting_stuck(drag):
    screens = [(0, 0, 100, 100), (900, 0, 100, 100)]
    backend = FakeInput()
    controller = Controller(Settings(), (0, 0, 1000, 100), backend, screens=screens)
    controller.resume()
    controller.machine.armed = True
    controller.mapper.filter.reset((50, 50))
    controller.target = (50, 50)
    if drag:
        controller.machine.down = True
        controller.drag_origin = controller.mapper.target((.5, .5), controller.settings)
        controller.drag_cursor = (50, 50)
        controller.drag_started = True
    for i in range(200):
        controller.process(Features((1, .5), .1 if drag else .8, .8, False), i * .03)
    positions = [event[1] for event in backend.events if event[0] == 'move']
    assert positions
    assert all(any(x <= px <= x+w-1 and y <= py <= y+h-1 for x,y,w,h in screens)
               for px, py in positions)
    assert positions[0][0] <= 99
    assert positions[-1][0] >= 900
    # Well past the drag grace period, so this is a lost hand rather than a
    # dropped frame and the pointer really does have to reacquire.
    controller.process(None, 6.4)
    controller.process(Features((1, .5), .8, .8, False), 6.43)
    assert controller.mapper.filter.value == controller.target
    if drag:
        assert backend.events[-1] == ('up',)
