import glob
import os
import sys
from typing import List

import cv2
import mediapipe as mp
import numpy as np

import dataset
from dataset import ModelLabel

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


def images_to_landmarks(image_paths: List[str], model_label: ModelLabel):
    total_hand_landmarks = []

    for path in image_paths:
        frame = cv2.imread(path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hand_landmarks = [[coord.x, coord.y, coord.z] for coord in hand_landmarks.landmark]
                total_hand_landmarks.append(hand_landmarks)

    np.save(dataset.get_dataset_path(model_label), total_hand_landmarks)


def create_data_dataset(model_label: ModelLabel, image_dataset_path):
    image_paths = glob.glob(os.path.join(image_dataset_path, '*'))
    images_to_landmarks(image_paths, model_label)


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} OK_DATASET_PATH NOT_OK_DATASET_PATH')
        return

    ok_dataset_path = sys.argv[1]
    not_ok_dataset_path = sys.argv[2]

    dataset.create_dir()
    create_data_dataset(ModelLabel.OK, ok_dataset_path)
    create_data_dataset(ModelLabel.NOT_OK, not_ok_dataset_path)


if __name__ == '__main__':
    main()
