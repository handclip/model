import os
from enum import Enum

import numpy as np

_DATA_DATASET_DIR_PATH = os.path.join(os.getcwd(), 'datasets', 'data')


class ModelLabel(Enum):
    OK = 'ok'
    NOT_OK = 'not_ok'


def get_dataset_path(label: ModelLabel):
    if label == ModelLabel.OK:
        return os.path.join(_DATA_DATASET_DIR_PATH, 'ok_landmarks.npy')
    elif label == ModelLabel.NOT_OK:
        return os.path.join(_DATA_DATASET_DIR_PATH, 'not_ok_landmarks.npy')
    else:
        raise ValueError('Invalid label')


def create_dir():
    os.makedirs(_DATA_DATASET_DIR_PATH, exist_ok=True)


def load(label: ModelLabel, flatten: bool = True) -> np.ndarray:
    dataset = np.load(get_dataset_path(label))
    return dataset.reshape(len(dataset), -1) if flatten else dataset
