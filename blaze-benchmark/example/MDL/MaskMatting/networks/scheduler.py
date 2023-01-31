from torch.optim.lr_scheduler import _LRScheduler

__all__ = [
        'GradualWarmupScheduler',
        'IsometryScheduler',
        'ExponentialScheduler',
        ]

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                    self.after_scheduler.last_epoch += 1
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class IsometryScheduler(_LRScheduler):
    """ Learning rate increase with an equal distance.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
    """

    def __init__(self, optimizer, multiplier, total_epoch, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        return super(IsometryScheduler, self).step(epoch)

class ExponentialScheduler(_LRScheduler):
    """ Learning rate decay exponentially.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_epoch: max iter
        exponent: power
    """

    def __init__(self, optimizer, max_epoch, exponent=0.9, last_epoch=-1):
        self.exponent = exponent
        self.max_epoch = max_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        multiplier = ((1 - float(self.last_epoch) / self.max_epoch) ** (self.exponent))
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):
        return super(ExponentialScheduler, self).step(epoch)