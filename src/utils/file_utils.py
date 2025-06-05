from model.config import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import json


def save_experiment_results(exp_dir, config_dict, train_history, val_history):
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as fp:
        json.dump(config_dict, fp, indent=2)

    df_hist = pd.DataFrame({
        "epoch": list(range(1, len(train_history) + 1)),
        "train_loss": train_history,
        "val_score": val_history
    })
    df_hist.to_csv(os.path.join(exp_dir, "histories.csv"), index=False)
