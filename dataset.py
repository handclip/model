import os
from enum import Enum
from typing import List

import numpy as np

_DATA_DATASET_DIR_PATH = os.path.join(os.getcwd(), 'datasets', 'data')


class ModelClass(Enum):
    OK = 'ok'
    NOT_OK = 'not_ok'


def save_data(model_class: ModelClass, total_hand_landmarks: List[List[List[float]]]):
    np.save(get_dataset_path(model_class), total_hand_landmarks)


def get_dataset_path(model_class: ModelClass):
    if model_class == ModelClass.OK:
        return os.path.join(_DATA_DATASET_DIR_PATH, 'ok_landmarks.npy')
    elif model_class == ModelClass.NOT_OK:
        return os.path.join(_DATA_DATASET_DIR_PATH, 'not_ok_landmarks.npy')
    else:
        raise ValueError('Invalid class')


def create_dir():
    os.makedirs(_DATA_DATASET_DIR_PATH, exist_ok=True)


def load(model_class: ModelClass, flatten: bool = True) -> np.ndarray:
    dataset = np.load(get_dataset_path(model_class))
    return dataset.reshape(len(dataset), -1) if flatten else dataset
