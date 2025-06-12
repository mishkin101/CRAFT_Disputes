from model.config import *
import matplotlib.pyplot as plt
import gdown
import numpy as np
import pandas as pd
import os
import sys
import json


def save_experiment_results_train(train_history, val_history):
    df_hist = {
        "epoch": list(range(1, len(train_history) + 1)),
        "train_loss": train_history,
        "val_score": val_history
    }
    with open(out_path, "w") as f:
    json.dump(df_hist, f, indent=2)

"""Handle logic for saving CRAFT model"""
def save_experiment_model(craft_model, loss, epoch):
      torch.save({
                    'epoch': epoch,
                    'en': craft_model.encoder.state_dict(),
                    'ctx': craft_model.context_encoder.state_dict(),
                    'atk_clf': craft_model.attack_clf.state_dict(),
                    'en_opt': craft_model.encoder_optimizer.state_dict(),
                    'ctx_opt': craft_model.context_encoder_optimizer.state_dict(),
                    'atk_clf_opt': craft_model.attack_clf_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': craft_model.voc.__dict__,
                    'embedding': craft_model.embedding.state_dict()
                }, os.path.join(experiment_model_dir, f"best_epoch_{epoch}.tar"))
      
def build_exp_directories(exp_name):
    if k_folds >1:
        for fold_idx, (train_ids, val_ids) in enumerate(folds, start=1):
            fold_name = f"fold_{fold_idx}"
            fold_dir = os.path.join(experiments_dir, fold_name)
            os.makedirs(fold_dir, exist_ok=True)
            model_dir   = os.path.join(fold_dir, "models")
            train_dir   = os.path.join(fold_dir, "training")
            results_dir = os.path.join(fold_dir, "results")
            plots_dir   = os.path.join(fold_dir, "plots")
            config_dir  = os.path.join(fold_dir, "config")
        else:
            model_dir = os.path.join(experiments_dir, experiment_name, "models")
            train_dir = os.path.join(experiments_dir, experiment_name, "training")
            results_dir = os.path.join(experiments_dir, experiment_name, "results")
            config_dir = os.path.join(experiments_dir, experiment_name, "config")
            plots_dir = os.path.join(experiments_dir, experiment_name, "plots")
    
        for d in (model_dir, train_dir, results_dir, plots_dir, config_dir):
            os.makedirs(d, exist_ok=True)