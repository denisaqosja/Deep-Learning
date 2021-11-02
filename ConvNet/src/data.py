from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Dataset:
    def __init__(self):
        self.BS = 256
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    def train_data(self):
        data = datasets.SVHN("./data/", split="train", download=True, transform=self.transform)
        trainLoader = DataLoader(data, batch_size=self.BS, shuffle=True)
        return trainLoader

    def test_data(self):
        data = datasets.SVHN("./data/", split="test", download=True, transform=transforms.ToTensor())
        testLoader = DataLoader(data, batch_size=self.BS, shuffle=False)
        return testLoader