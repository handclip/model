import glob
import os
import sys
from typing import List

import cv2
import mediapipe as mp

import dataset
from dataset import ModelClass

hands = mp.solutions.hands.Hands()


def convert_imgs_to_data(img_paths: List[str]):
    total_hand_landmarks = []

    for path in img_paths:
        frame = cv2.imread(path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hand_landmarks = [(coord.x, coord.y, coord.z)
                                  for coord in hand_landmarks.landmark]
                total_hand_landmarks.append(hand_landmarks)

    return total_hand_landmarks


def create_data_dataset(model_class: ModelClass, img_dataset_path: str):
    img_paths = glob.glob(os.path.join(img_dataset_path, '*'))
    data_dataset = convert_imgs_to_data(img_paths)
    dataset.save_data(model_class, data_dataset)


def main():
    create_data_dataset(ModelClass.OK, dataset.get_img_paths(ModelClass.OK))
    create_data_dataset(ModelClass.NOT_OK,
                        dataset.get_img_paths(ModelClass.NOT_OK))


if __name__ == '__main__':
    main()
