import torch.nn
import torch.optim

def load_optimizer(optimizerName, parameters, lr):
    if optimizerName == "SGD":
        optimizer = torch.optim.SGD(parameters, lr)
    elif optimizerName == "RMSprop":
        optimizer = torch.optim.RMSprop(parameters, lr)
    elif optimizerName == "Adam":
        optimizer = torch.optim.Adam(parameters, lr)
    else:
        optimizer = None

    return optimizer

