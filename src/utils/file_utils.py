from model.config import *
import matplotlib.pyplot as plt
import gdown
import numpy as np
import pandas as pd
import os
import sys
import json

"""Handle logic for saving training metrics"""
def save_experiment_results_train(out_dir, train_dict):
    train_dir = os.path.join(out_dir, "train_metrics.txt")
    with open(train_dir, "w") as f:
        json.dump(train_dict, f, indent=2)


"""Handle logic for saving CRAFT model"""
def save_experiment_model(craft_model):
      torch.save({
                    'en': craft_model.encoder.state_dict(),
                    'ctx': craft_model.context_encoder.state_dict(),
                    'atk_clf': craft_model.attack_clf.state_dict(),
                    'en_opt': craft_model.encoder_optimizer.state_dict(),
                    'ctx_opt': craft_model.context_encoder_optimizer.state_dict(),
                    'atk_clf_opt': craft_model.attack_clf_optimizer.state_dict(),
                    'voc_dict': craft_model.voc.__dict__,
                    'embedding': craft_model.embedding.state_dict()
                }, os.path.join(experiments_dir, experiment_name, f"model_{experiment_name}.tar"))
      
"""Handle logic for creating fold experiment results"""
def build_fold_directories(fold_idx):
        fold_name = f"fold_{fold_idx}"
        fold_dir = os.path.join(experiments_dir, experiment_name, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        model_dir   = os.path.join(fold_dir, "models")
        train_dir   = os.path.join(fold_dir, "training")
        results_dir = os.path.join(fold_dir, "results")
        plots_dir   = os.path.join(fold_dir, "plots")
        config_dir  = os.path.join(fold_dir, "config")
    
        for d in (model_dir, train_dir, results_dir, plots_dir, config_dir):
            os.makedirs(d, exist_ok=True)
        return model_dir, train_dir, results_dir, plots_dir, config_dir