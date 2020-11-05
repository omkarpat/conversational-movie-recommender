
class Metric(object):
    def get(self):
        raise NotImplementedError("Subclasses must implement this")

class RunningMetric(Metric):
    def __init__(self):
        self.current_value = 0.0
        self.num_steps = 0

    def _add(self, value):
        self.current_value += (value - self.current_value) / (self.num_steps + 1)
        self.num_steps += 1

    def add(self, value):
        self._add(value)

    def get(self):
        return self.current_value