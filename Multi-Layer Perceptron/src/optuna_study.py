import torch
import torch.nn as nn
import torch.optim as optim

import optuna
import joblib
import pickle

from model import MLP
from train import Trainer

def objective(trial):

    model = MLP()

    optimizers = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    optimizer = getattr(optim, optimizers)(model.parameters(), lr)

    trainer = Trainer()
    trainer.setup_model()
    trainer.optimizer = optimizer
    trainer.train_model()

    accuracy = trainer.valid_acc

    return accuracy


def load_best_hyperParams():
    with open("study.pkl", "rb") as file:
        study = pickle.load(file)

    print(study.best_trial.params)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    with open("study.pkl", "wb") as file:
        pickle.dump(study, file)

    load_best_hyperParams()