import os
import torch
import mlflow
import ray.tune as tune
from ray.tune import CLIReporter
from ray.air.config import RunConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback

from runners.finetune import finetune_craft

from model.data import *



def get_default_config():
    # Assuming config.py defines these variables:
    default_cfg = {
        "dropout": dropout,
        "batch_size": batch_size,
        "clip": clip,
        "learning_rate": learning_rate,
        "print_every": print_every,
        "finetune_epochs": finetune_epochs,
        "validation_size": val_size,
        "ray_tune": True,  # enable reporting inside finetune_craft
        "optimizer_type": optimizer_type,
        "loss_function": loss_function,
        "activation": activation,
        "k_folds": k_folds,
        "last_only_train": last_only_train,
        "last_only_val": last_only_val,
        # add any other custom config variables from config.py
    }
    return default_cfg


search_space = {
  **get_default_config(),
  "dropout":          tune.grid_search([0.1, 0.2, 0.3]),
  "batch_size":       tune.choice([2**i for i in range(2, 9)]),
  "clip":             tune.grid_search([50.0, 100.0]),
  "learning_rate":    tune.loguniform([1e-5, 5e-5, 1e-4]),
  "finetune_epochs":  tune.grid_search([10,20,30])
  # you can sweep any other things from DEFAULT_CONFIG in the same way…
}
epochs_choices = search_space["finetune_epochs"].categories
max_t = max(epochs_choices)

scheduler = ASHAScheduler(
    max_t=max_t,     # the “T” in ASHA is your number of epochs
    grace_period=5,            # minimum epochs before stopping a bad trial
    reduction_factor=2,        # how aggressively to stop
    brackets=1
)

trainable_wrapped = tune.with_resources(
    finetune_craft,
    {"cpu": 4, "gpu": 1}
)


tune_config = TuneConfig(
    metric="accuracy",
    mode="max",
    max_concurrent_trials=10,
    num_samples=1,
    search_alg=None
)

runConfig = RunConfig(
        name="craft_hpo",
        storage_path="~/ray_results",
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="file:///your/mlflow/artifact/dir",
                experiment_name="craft_experiment"
            )   
        ],
    )

tuner = tune.Tuner(
    tune.with_resources(
        finetune_craft,
        resources={"cpu": 4, "gpu": 1},
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        metric="accuracy",        # maximize validation accuracy
        mode="max",
        num_samples=20,             # number of hyperparameter trials
    ),
    param_space=search_space,
)


results = tuner.fit()
best = results.get_best_result(metric="mean_accuracy", mode="max")
print("Best config:", best.config)
print("Best mean_accuracy:", best.metrics["mean_accuracy"])

