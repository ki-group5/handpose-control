
import cv2
import numpy as np

from utils.render import PoseRender
from utils.normalized import NormalizedData
from vec_math import Quat


def example_normalization():
    """
    Just a main to show the normalization
    """
    import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    # mp_hands = mp.solutions.hands
    mp_holistic = mp.solutions.holistic

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = holistic.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # mp_drawing.draw_landmarks(
            #     image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # mp_drawing.draw_landmarks(
            #     image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


            if results.left_hand_landmarks:
                left = NormalizedData.create(results.left_hand_landmarks.landmark, "Left")
                PoseRender.draw_normal(left, image)

            if results.right_hand_landmarks:
                right = NormalizedData.create(results.right_hand_landmarks.landmark, "Right")
                PoseRender.draw_normal(right, image)


            cv2.imshow('MediaPipe Holistic', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    example_normalization()
