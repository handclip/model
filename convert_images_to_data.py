import glob
import os
import sys
from typing import List

import cv2
import mediapipe as mp

import dataset
from dataset import ModelClass

hands = mp.solutions.hands.Hands()


def convert_images_to_data(image_paths: List[str]):
    total_hand_landmarks = []

    for path in image_paths:
        frame = cv2.imread(path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hand_landmarks = [(coord.x, coord.y, coord.z)
                                  for coord in hand_landmarks.landmark]
                total_hand_landmarks.append(hand_landmarks)

    return total_hand_landmarks


def create_data_dataset(model_class: ModelClass, image_dataset_path: str):
    image_paths = glob.glob(os.path.join(image_dataset_path, '*'))
    data_dataset = convert_images_to_data(image_paths)
    dataset.save_data(model_class, data_dataset)


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} OK_DATASET_PATH NOT_OK_DATASET_PATH')
        return

    ok_dataset_path = sys.argv[1]
    not_ok_dataset_path = sys.argv[2]

    dataset.create_dir()
    create_data_dataset(ModelClass.OK, ok_dataset_path)
    create_data_dataset(ModelClass.NOT_OK, not_ok_dataset_path)


if __name__ == '__main__':
    main()
