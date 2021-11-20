import cv2
import mediapipe as mp

import dataset
from dataset import ModelClass

hands = mp.solutions.hands.Hands()


def video_to_landmarks(video_path):
    total_hand_landmarks = []

    cap = cv2.VideoCapture(video_path)
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


def main():
    ok_left_data = video_to_landmarks('datasets/vids/ok_left.mov')
    ok_right_data = video_to_landmarks('datasets/vids/ok_right.mov')
    not_ok_left_data = video_to_landmarks('datasets/vids/not_ok_left.mov')
    not_ok_right_data = video_to_landmarks('datasets/vids/not_ok_right.mov')

    dataset.save_data(ModelClass.OK, ok_left_data + ok_right_data)
    dataset.save_data(ModelClass.NOT_OK, not_ok_left_data + not_ok_right_data)


if __name__ == '__main__':
    main()
