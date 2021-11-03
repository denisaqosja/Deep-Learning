import pickle
import optuna

from model import CNN
from trainer import Trainer
from utils import set_optimizer

def objective(trial):

    model = CNN()
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "SGD_momentum", "RMSprop", "Adam"])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    regularizer = trial.suggest_categorical("regularizer", ["L1", "L2", "Elastic", "None"])

    trainer = Trainer()
    trainer.setup_model()
    trainer.optimizer = set_optimizer(optimizer, model.parameters(), learning_rate)
    trainer.lr = learning_rate
    trainer.regularizerName = regularizer
    trainer.train_model()

    accuracy = trainer.test_acc

    return accuracy


if __name__=="__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials = 100)

    with open("optuna_study.pkl", "wb") as file:
        pickle.dump(study, file)