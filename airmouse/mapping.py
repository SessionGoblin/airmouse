import math

class AdaptiveEMA:
    def __init__(self):
        self.value = None

    def reset(self, point=None):
        self.value = point

    def update(self, point, dt, tau, deadzone):
        if self.value is None:
            self.value = point
        distance = math.dist(point, self.value)
        if distance > deadzone:
            alpha = 1 - math.exp(-max(.001, dt) / (tau / (1 + distance / 70)))
            self.value = tuple(a + alpha * (b-a) for a, b in zip(self.value, point))
        return self.value

class CursorMapper:
    def __init__(self, bounds, screens=None):
        self.bounds = bounds
        self.screens = tuple(screens) if screens else (bounds,)
        self.filter = AdaptiveEMA()

    def visible_point(self, point):
        """Project a desktop point onto the nearest actual monitor rectangle."""
        candidates = [(min(x+w-1, max(x, point[0])),
                       min(y+h-1, max(y, point[1])))
                      for x, y, w, h in self.screens]
        return min(candidates, key=lambda candidate: math.dist(point, candidate))

    def target(self, point, settings):
        x, y, width, height = self.bounds
        def normalized(v):
            return min(1, max(0, (((v-settings.margin)/(1-2*settings.margin))-.5)*settings.sensitivity+.5))
        return x + normalized(point[0])*(width-1), y + normalized(point[1])*(height-1)

    def update(self, point, settings, dt):
        target = self.target(point, settings)
        result = self.filter.update(target, dt, settings.smoothing, settings.deadzone)
        x, y, w, h = self.bounds
        # Keep subpixel progress; only the input boundary should round pixels.
        return min(x+w-1, max(x, result[0])), min(y+h-1, max(y, result[1]))
