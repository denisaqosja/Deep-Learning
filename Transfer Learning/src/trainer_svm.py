"""
ResNet for feature extraction + SVM for classification
"""

import torch
import sklearn
from sklearn import svm

from data import Dataset
from model import ResNet
from utils import set_model

class Trainer_SVM():
    def __init__(self):
        pass

    def load_data(self):
        datasets = Dataset()
        self.train_loader, self.test_loader = datasets.data_loader()

        #prepare the test dataset
        """
        self.test_dataset_images = []
        self.test_dataset_labels = []

        for id, (images, labels) in enumerate(self.test_loader):
            for image in images:
                #print(image.shape)
                self.test_dataset_images.append(image.numpy())
            for label in labels:
                self.test_dataset_labels.append(label.numpy())

        return
        """

    def resnet_featureExtractor(self, data_loader):

        activations_list = []
        labels_list = []

        m = ResNet(fixed=True)
        self.model = m.model(svm_wanted=True)

        for batchId, (images, labels) in enumerate(data_loader):
            outputs = self.model(images)
            for _, activation_map in enumerate(outputs):
                activations_list.append(activation_map.numpy())
            for label in labels:
                labels_list.append(label.numpy())

        return activations_list, labels_list

    def svm_classifier(self):
        self.load_data()
        svm_cl = svm.SVC(kernel="rbf")

        #Training step
        activations, labels = self.resnet_featureExtractor(self.train_loader)
        svm_cl.fit(activations, labels)

        #Training step: predict new data
        activations_test, labels_test = self.resnet_featureExtractor(self.test_loader)
        predictions = svm_cl.predict(activations_test)

        accuracy = sklearn.metrics.accuracy_score(y_true=labels_test, y_pred=predictions)

        print(f"Accuracy of ResNet + SVM: {accuracy}")
        print(f"Precision: {sklearn.metrics.precision_score(y_true=labels_test, y_pred=predictions)}")
        print(f"Recall: {sklearn.metrics.recall_score(y_true=labels_test, y_pred=predictions)}")

if __name__ == "__main__":
    trainer = Trainer_SVM()
    trainer.svm_classifier()