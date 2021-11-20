import cv2
import mediapipe as mp

import dataset
from dataset import ModelClass

hands = mp.solutions.hands.Hands()


def vid_to_landmarks(vid_path):
    total_hand_landmarks = []

    cap = cv2.VideoCapture(vid_path)
    while True:
        _, frame = cap.read()

        if frame is None:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hand_landmarks = [(coord.x, coord.y, coord.z)
                                  for coord in hand_landmarks.landmark]
                total_hand_landmarks.append(hand_landmarks)

    cap.release()
    cv2.destroyAllWindows()
    return total_hand_landmarks


def save_landmarks(model_class: ModelClass):
    vid_paths = dataset.get_vids_paths(model_class)
    data = [vid_to_landmarks(path) for path in vid_paths]
    dataset.save_data(model_class, data)


def main():
    save_landmarks(ModelClass.OK)
    save_landmarks(ModelClass.NOT_OK)


if __name__ == '__main__':
    main()
