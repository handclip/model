import glob
import json
import os
import sys
from enum import Enum

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


DATA_DATASET_DIR_PATH = os.path.join(os.getcwd(), 'datasets', 'data')


class ModelLabel(Enum):
    OK = 'ok'
    NOT_OK = 'not_ok'


def images_to_landmarks(image_paths):
    total_hand_landmarks = []

    for path in image_paths:
        frame = cv2.imread(path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                coords = [[coord.x, coord.y, coord.z] for coord in hand_landmarks.landmark]
                total_hand_landmarks.append(coords)

    return total_hand_landmarks


def save_landmarks(landmarks, file_name):
    with open(os.path.join(DATA_DATASET_DIR_PATH, file_name), 'w') as f:
        f.write(json.dumps(landmarks))


def create_data_dataset(model_label: ModelLabel, image_dataset_path):
    image_paths = glob.glob(os.path.join(image_dataset_path, '*'))
    landmarks = images_to_landmarks(image_paths)
    save_landmarks(landmarks, f'{model_label.value}_landmarks.json')


def main():
    if len(sys.argv) != 3:
        print(f'usage: {sys.argv[0]} OK_DATASET_PATH NOT_OK_DATASET_PATH')
        return

    ok_dataset_path = sys.argv[1]
    not_ok_dataset_path = sys.argv[2]

    os.makedirs(DATA_DATASET_DIR_PATH, exist_ok=True)
    create_data_dataset(ModelLabel.OK, ok_dataset_path)
    create_data_dataset(ModelLabel.NOT_OK, not_ok_dataset_path)


if __name__ == '__main__':
    main()
