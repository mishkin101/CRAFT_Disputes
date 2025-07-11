from .config import *
from .data import *
from .model import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam


class OptimizerWithScheduler:
    def __init__(self, optimizer, models):
        self.models = models
        self.optimizer = optimizer
        if scheduling == True:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode = mode, 
                factor=factor,
                patience=patience,
                threshold=threshold,
            )
        else:
            self.scheduler = None
        self.clip = clip
    

    def batchStep(self, loss):
        print("updating loss")
        """Call this once per training batch."""
        self.optimizer.zero_grad()
        loss.backward()
        all_params = []
        for m in self.models:
            all_params += list(m.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, self.clip)
        self.optimizer.step()

    def epochStep(self, epoch_val_score):
        print("scheduler step for learning rate adjustment")
        """Call this once per validation epoch."""
        if self.scheduler is not None:
            self.scheduler.step(epoch_val_score)
            print(f"learning rate is: {self.scheduler.get_last_lr()}")