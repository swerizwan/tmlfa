# Import necessary libraries
import torch
from torch.optim.lr_scheduler import _LRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_scheduler(optimizer, lr_name, warmup_lr, min_lr, num_steps, warmup_steps, decay_steps=100, decay_rate=0.1):
    """
    Factory function to create a learning rate scheduler based on the specified type.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used with the scheduler.
        lr_name (str): The type of learning rate scheduler to create ('cosine', 'linear', 'step').
        warmup_lr (float): The initial learning rate during the warmup phase.
        min_lr (float): The minimum learning rate for the 'cosine' scheduler.
        num_steps (int): The total number of steps for the scheduler.
        warmup_steps (int): The number of steps for the warmup phase.
        decay_steps (int, optional): The interval steps for learning rate decay in the 'step' scheduler. Defaults to 100.
        decay_rate (float, optional): The decay rate for the 'step' scheduler. Defaults to 0.1.

    Returns:
        _LRScheduler: The created learning rate scheduler.
    """
    lr_scheduler = None
    if lr_name == 'cosine':
        # Create a cosine learning rate scheduler
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            cycle_mul=1.,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif lr_name == 'linear':
        # Create a linear learning rate scheduler
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif lr_name == 'step':
        # Create a step learning rate scheduler
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    """
    A custom linear learning rate scheduler.

    This scheduler linearly decreases the learning rate from the initial value to a fraction of the initial value.
    It also supports a warmup phase where the learning rate increases linearly from `warmup_lr_init` to the initial value.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        """
        Initialize the LinearLRScheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to be used with the scheduler.
            t_initial (int): The total number of steps for the scheduler.
            lr_min_rate (float): The minimum learning rate as a fraction of the initial learning rate.
            warmup_t (int, optional): The number of warmup steps. Defaults to 0.
            warmup_lr_init (float, optional): The initial learning rate during the warmup phase. Defaults to 0.
            t_in_epochs (bool, optional): Whether the steps are in epochs or iterations. Defaults to True.
            noise_range_t (int, optional): The range of steps for adding noise to the learning rate. Defaults to None.
            noise_pct (float, optional): The percentage of noise to add to the learning rate. Defaults to 0.67.
            noise_std (float, optional): The standard deviation of the noise. Defaults to 1.0.
            noise_seed (int, optional): The seed for the noise. Defaults to 42.
            initialize (bool, optional): Whether to initialize the scheduler. Defaults to True.
        """
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            # Calculate the step size for the warmup phase
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        """
        Calculate the learning rate at the given step.

        Args:
            t (int): The current step.

        Returns:
            list: A list of learning rates for each parameter group.
        """
        if t < self.warmup_t:
            # During warmup, linearly increase the learning rate
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            # After warmup, linearly decrease the learning rate
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        """
        Get the learning rate values for the given epoch.

        Args:
            epoch (int): The current epoch.

        Returns:
            list: A list of learning rates for each parameter group, or None if not in epochs.
        """
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        """
        Get the learning rate values for the given number of updates.

        Args:
            num_updates (int): The current number of updates.

        Returns:
            list: A list of learning rates for each parameter group, or None if not in iterations.
        """
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None