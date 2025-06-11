from model.config import *
from model.data import *
from model.model import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam


class OptimizerWithScheduler:
    def __init__(self, optimizer, models):
        self.models = models
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode = mode, 
            factor=factor,
            patience=patience,
            threshold=threshold,
        )
        self.clip = clip
    

    def batchStep(self, loss):
        """Call this once per training batch."""
        for model in self.models:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optimizer.parameters(), self.clip)
            self.optimizer.step()

    def epochStep(self, val_score):
        """Call this once per validation epoch."""
        for model in self.models:
            self.scheduler.step(val_score)