
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


class RunningLambdaMetric(RunningMetric):
    def __init__(self, fn):
        self.func = fn
        super().__init__()

    def add(self, *args, **kwargs):
        self._add(float(self.func(*args, **kwargs)))

class MetricLambda(Metric):
    def __init__(self, fn, metric):
        self.func = fn
        self.metric = metric

    def get(self):
        return self.func(self.metric.get())

class Accuracy(RunningLambdaMetric):
    def __init__(self):
        super().__init__(self._compute_accuracy)

    def _compute_accuracy(self, logits, labels):
        predictions = logits.argmax(dim=-1)

        return (predictions == labels).mean()
