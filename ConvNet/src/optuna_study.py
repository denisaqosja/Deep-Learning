import pickle
import optuna

from model import CNN
from train import Trainer
from utils import set_optimizer

def objective(trial):

    model = CNN()
    optimizer = trial.suggest_categorical("SGD", "SGD_momentum", "RMSprop", "Adam")
    learning_rate = trial.suggest_loguniform(1e-5, 5e-2)
    regularizer = trial.suggest_categorical("L1", "L2", "Elastic", "None")

    trainer = Trainer()
    trainer.setup_model()
    trainer.optimizer = set_optimizer(optimizer, model.parameters(), learning_rate)
    trainer.lr = learning_rate
    trainer.regularizer = regularizer

    accuracy = trainer.eval_acc

    return accuracy


if __name__=="__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective)

    with open("optuna_study.pkl", "wb") as file:
        pickle.dump(file)