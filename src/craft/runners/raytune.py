import os
import ray
import ray.tune as tune
from ray.tune import CLIReporter
from ray.tune import RunConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback
from types import ModuleType
import craft.utils.finetune_utils as finetune_utils

import craft.model.config as cfg_mod


def get_config_dict():
    out = {}
    for name, val in vars(cfg_mod).items():
        if name.startswith("_"):
            continue
        if name in cfg_mod._CONFIG_BLACKLIST:
            continue
        if isinstance(val, ModuleType) or callable(val):
            continue
        if isinstance(val, (int, float, bool, str, list, dict, tuple)):
            try:
                out[name] = val
            except (TypeError, ValueError):
                out[name] = str(val)
    return out



search_space = {
  **get_config_dict(),
  "dropout":          tune.grid_search([0.1, 0.2, 0.3]),
  "batch_size":       tune.choice([2**i for i in range(2, 9)]),
  "clip":             tune.loguniform(1.0, 100.0),
  "learning_rate":    tune.loguniform(1e-5, 1e-1),
  "finetune_epochs":  tune.grid_search([10,20,30])
  # you can sweep any other things from DEFAULT_CONFIG in the same way…
}

search_space_test = {
    **get_config_dict(),
    # trivial sweep: only one choice each
    "dropout": tune.grid_search([0.1]),
    "batch_size": tune.choice([8]),
    "clip": tune.choice([1.0]),
    "learning_rate": tune.choice([1e-3]),
    "finetune_epochs": tune.grid_search([1]),
}

max_iter = search_space["finetune_epochs"]['grid_search'][-1]
scheduler = ASHAScheduler(
    max_t= max_iter,     # the “T” in ASHA is your number of epochs
    grace_period=5,            # minimum epochs before stopping a bad trial
    reduction_factor=2,        # how aggressively to stop
    brackets=1,
    metric = "mean_val_f1_micro",
    mode ="max"
)


tune_config = TuneConfig(
    # metric= "mean_val_f1_score",
    # mode="max",
    scheduler = scheduler,
    max_concurrent_trials=10,
    num_samples=1,
    search_alg=None,
)

runConfig = RunConfig(
        name="craft_hpo",
        storage_path="/Users/mishkin/Desktop/Research/CRAFT_Disputes/CRAFT_Disputes/ray_results",
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="http://0.0.0.0:8080",
                experiment_name="craft_experiment_raytune_test"
            )   
        ],
    )




tuner = tune.Tuner(
    tune.with_resources(finetune_utils.finetune_craft_test,
        resources={"cpu": 2, "gpu": 0}),
    tune_config=tune_config,
    param_space=search_space_test,
    run_config=runConfig
)


def run_tune():
 return

if __name__ == "__main__":
    ray.init( 
        address="auto", 
    runtime_env={
      "working_dir": "/Users/mishkin/Desktop/Research/CRAFT_Disputes/CRAFT_Disputes/src/craft", 
      "excludes": [".git", ".git/*", "src/craft/experiments/*", "mlruns/*", "ray_results/*", "data/*", "saved_models/*"]
        }
    )
    results = tuner.fit()
    df = results.get_dataframe()
    print(df.head())
    best = results.get_best_result(metric="mean_val_f1_micro", mode="max")
    print("Best config:", best.config)
    print("Best mean_accuracy:", best.metrics["mean_val_accuracy"])