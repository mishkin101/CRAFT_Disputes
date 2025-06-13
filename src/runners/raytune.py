import os
import ray.tune as tune
from ray.tune import CLIReporter
from ray.air.config import RunConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback
from types import ModuleType
from utils.finetune_utils import finetune_craft
import model.config as cfg_mod


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
max_iter = search_space["finetune_epochs"]['grid_search'][-1]
scheduler = ASHAScheduler(
    max_t= max_iter,     # the “T” in ASHA is your number of epochs
    grace_period=5,            # minimum epochs before stopping a bad trial
    reduction_factor=2,        # how aggressively to stop
    brackets=1,
    metric = "mean_val_accuracy",
    mode ="max"
)


tune_config = TuneConfig(
    metric= "mean_val_accuracy",
    mode="max",
    scheduler =scheduler,
    max_concurrent_trials=10,
    num_samples=2,
    search_alg=None,
)

runConfig = RunConfig(
        name="craft_hpo",
        storage_path="~/ray_results",
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="http://127.0.0.1:5000",
                experiment_name="craft_experiment"
            )   
        ],
    )

tuner = tune.Tuner(
    tune.with_resources(
        finetune_craft,
        resources={"cpu": 4, "gpu": 1},
    ),
    tune_config=tune_config,
    param_space=search_space,
)


if __name__ == "__main__":
    results = tuner.fit()
    best = results.get_best_result(metric="mean_accuracy", mode="max")
    print("Best config:", best.config)
    print("Best mean_accuracy:", best.metrics["mean_accuracy"])