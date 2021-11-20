import pickle

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import dataset
from dataset import ModelClass


def save_model(model):
    with open('model.pickle', 'wb') as model_file:
        model_file.write(pickle.dumps(model))


def eval_model(model, test_data, test_classes):
    predicted = model.predict(test_data)
    print(classification_report(test_classes, predicted))


def get_training_data():
    ok_dataset = dataset.load(ModelClass.OK)
    not_ok_dataset = dataset.load(ModelClass.NOT_OK)
    data = np.concatenate((ok_dataset, not_ok_dataset))
    classes = np.concatenate(
        (np.repeat(0, len(ok_dataset)), np.repeat(1, len(not_ok_dataset))))
    return data, classes


def main():
    classifier = svm.SVC()
    data, classes = get_training_data()
    train_data, test_data, train_classes, test_classes = train_test_split(
        data, classes, test_size=0.2)
    classifier.fit(train_data, train_classes)

    eval_model(classifier, test_data, test_classes)
    save_model(classifier)


if __name__ == '__main__':
    main()
