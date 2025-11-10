from torch.optim.lr_scheduler import _LRScheduler


class CustomStepCycleLR(_LRScheduler):
    def __init__(self, optimizer, steps, step_size, last_epoch=-1):
        self.steps = steps
        self.stap_size = step_size
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        step_idx = (step // self.stap_size) % len(self.steps)
        lrs = [float(self.steps[step_idx])]
        return lrs
