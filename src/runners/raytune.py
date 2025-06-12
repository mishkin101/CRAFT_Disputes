import os
import torch
import mlflow
import ray.tune as tune
from tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback

from runners.finetune import main, loadDataset, loadPretrainedModel, loadCheckpoint
from runners.finetune import setOptimizer, setLossFunction, setPredictorActivation, CraftPipeline, computerIterations, trainIter

from model.data import *


search_space = {
  "dropout":          tune.grid_search([0.1, 0.2, 0.3]),
  "batch_size":       tune.grid_search([32, 64]),
  "clip":             tune.grid_search([50.0, 100.0]),
  "learning_rate":    tune.grid_search([1e-5, 5e-5, 1e-4]),
  # you can sweep any other things from DEFAULT_CONFIG in the same way…
}

asha = ASHAScheduler(
    max_t=finetune_epochs,     # the “T” in ASHA is your number of epochs
    grace_period=5,            # minimum epochs before stopping a bad trial
    reduction_factor=2,        # how aggressively to stop
    brackets=1
)

tuner = tune.Tuner(
    train_mnist,
    param_space=search_space,
)
results = tuner.fit()

def train_tune(config, checkpoint_dir=None):
    mlflow.start_run(run_name=tune.get_trial_id())
    mlflow.log_params({
        "learning_rate": config["learning_rate"],
        "batch_size":    config["batch_size"],
        "dropout":       config["dropout"],
        # …any other sweepable params
    })
    base_cfg = config["base_config"]
