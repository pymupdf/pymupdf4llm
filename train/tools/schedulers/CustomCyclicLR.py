from torch.optim.lr_scheduler import _LRScheduler


class CustomCyclicLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up, decay_factor=1.0, last_epoch=-1):
        """
        Custom CyclicLR with decaying max_lr per cycle.

        Args:
            optimizer (Optimizer): wrapped optimizer.
            base_lr (float): initial learning rate (lowest point of cycle).
            max_lr (float): initial max learning rate (peak of first cycle).
            step_size_up (int): number of steps to reach max_lr.
            decay_factor (float): multiplicative decay factor for max_lr after each cycle.
            last_epoch (int): the index of last epoch. Default: -1.
        """
        self.base_lr = base_lr
        self.initial_max_lr = max_lr
        self.step_size_up = step_size_up
        self.cycle_length = step_size_up * 2  # up + down
        self.decay_factor = decay_factor

        # 초기화 시 optimizer의 learning rate도 세팅
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        cycle_idx = step // self.cycle_length
        cycle_pos = step % self.cycle_length

        # 현재 cycle의 max_lr를 decay 적용해서 계산
        max_lr = self.initial_max_lr * (self.decay_factor ** cycle_idx)

        # 삼각형 형태의 상승/하강 곡선
        if cycle_pos < self.step_size_up:
            scale = cycle_pos / self.step_size_up  # 상승 phase
        else:
            scale = (2 - (cycle_pos / self.step_size_up))  # 하강 phase

        # 최종 learning rate = base_lr + (max_lr - base_lr) * scale
        lrs = [self.base_lr + (max_lr - self.base_lr) * scale for _ in self.optimizer.param_groups]
        return lrs
