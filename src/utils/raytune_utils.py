import optuna


def objective(trial):
    trial_dropout = trial.suggest_float("dropout", 0.0, 0.5)
    trial_lr = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    trial_batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    trial_clip = trial.suggest_float("clip", 1.0, 100.0)

