
from data.data_file import Label
from controller.command import Cmd, Commander
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from classifier.centroid.centroid_classifier import CentroidClassifier
from utils.render import PoseRender
from utils.normalized import NormalizedData
from vec_math import Quat



COMMANDS: Dict[str, List[Cmd]] = {
    "stop": [Cmd("flat", 0.9), Cmd("fist", 0.9), Cmd("flat", 0.9)],
    "continue": [Cmd("index", 0.9), Cmd("fist", 0.9)]
}



def example_normalization():
    """
    Just a main to show the normalization
    """
    import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    # mp_hands = mp.solutions.hands
    mp_holistic = mp.solutions.holistic


    # Pose classifier
    classifier = CentroidClassifier()

    # Command state machine from pose
    labels = [l.value for l in Label]

    commander_left = Commander(COMMANDS, labels)
    commander_right = Commander(COMMANDS, labels)
    

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as holistic:
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
                left = NormalizedData.create_from_mediapipe(results.left_hand_landmarks.landmark, "Left")

                # left_reconstr = left.reconstruct()
                # PoseRender.render_landmarks(image, left_reconstr, mp_holistic.HAND_CONNECTIONS)
                
                PoseRender.draw_normal(left, image)
                prediction = classifier.classify(left.direction)
                cv2.putText(image, f'Left: {prediction}', color=(255, 0, 0), org=(100, 150),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)


                commander_left.push(prediction)
            else:
                commander_left.push(None)

            if results.right_hand_landmarks:
                right = NormalizedData.create_from_mediapipe(results.right_hand_landmarks.landmark, "Right")

                # right_reconstr = right.reconstruct()
                # PoseRender.render_landmarks(image, right_reconstr, mp_holistic.HAND_CONNECTIONS)

                PoseRender.draw_normal(right, image)
                prediction = classifier.classify(right.direction)
                cv2.putText(image, f'Right: {prediction}', color=(255, 0, 0), org=(100, 100),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)

                commander_right.push(prediction)
            else:
                commander_left.push(None)

            # command_left = detect_commands(Commands, states_left)
            # if command_left:
            #     states_left.clear()
            #     print("Left command:", command_left)


            # command_right = detect_commands(Commands, states_right)
            # if command_right:
            #     states_left.clear()
            #     print("Right command:", command_right)

            cv2.imshow('MediaPipe Holistic', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    example_normalization()
