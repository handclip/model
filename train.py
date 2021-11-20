import pickle

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import dataset
from dataset import ModelLabel


def save_model(model):
    with open('model.pickle', 'wb') as f:
        f.write(pickle.dumps(model))


def eval_model(model, X_test, y_test):
    predicted = model.predict(X_test)
    classification_report(y_test, predicted)


def get_training_data():
    ok_dataset = dataset.load(ModelLabel.OK)
    not_ok_dataset = dataset.load(ModelLabel.NOT_OK)
    X = np.concatenate((ok_dataset, not_ok_dataset))
    y = np.concatenate((np.repeat(0, len(ok_dataset)), np.repeat(1, len(not_ok_dataset))))
    return X, y


def main():
    clf = svm.SVC()
    X, y = get_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)

    eval_model(clf, X_test, y_test)
    save_model(clf)


if __name__ == '__main__':
    main()
