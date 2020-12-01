import numbers
from collections import Sequence

from torch.optim.lr_scheduler import _LRScheduler

class PiecewiseLinearLR(_LRScheduler):
    """
    LR Scheduler that linearly decreases the LR over the
    duration of the number of steps, based off the Ignite
    implementation.

    I didn't want to change any of the original setup from TransferTransfo
    therefore, it felt more appropriate to retrofit a custom scheduler
    """

    def __init__(self, optimizer, milestones_values, last_epoch=-1):

        values = []
        milestones = []

        for pair in milestones_values:
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError("Argument milestones_values should be a list of pairs (milestone, param_value)")
            if not isinstance(pair[0], numbers.Integral):
                raise ValueError("Value of a milestone should be integer, but given {}".format(type(pair[0])))
            if len(milestones) > 0 and pair[0] < milestones[-1]:
                raise ValueError("Milestones should be increasing integers, but given {} is smaller "
                                 "than the previous milestone {}".format(pair[0], milestones[-1]))
            milestones.append(pair[0])
            values.append(pair[1])

        self.values = values
        self.milestones = milestones
        self._index = 0
        self.last_epoch = last_epoch
        super(PiecewiseLinearLR, self).__init__(optimizer, last_epoch)

    def _get_start_end(self):
        if self.milestones[0] > self.last_epoch:
            return self.last_epoch - 1, self.last_epoch, self.values[0], self.values[0]
        elif self.milestones[-1] <= self.last_epoch:
            return self.last_epoch, self.last_epoch + 1, self.values[-1], self.values[-1],
        elif self.milestones[self._index] <= self.last_epoch < self.milestones[self._index + 1]:
            return self.milestones[self._index], self.milestones[self._index + 1], \
                   self.values[self._index], self.values[self._index + 1]
        else:
            self._index += 1
            return self._get_start_end()

    def get_lr(self):
        start_index, end_index, start_value, end_value = self._get_start_end()
        return [start_value + (end_value - start_value) * (self.last_epoch - start_index) / (end_index - start_index)]
