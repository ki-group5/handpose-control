"""

Data Recorder

Run with in project folder:
```
python3 data/record.py Fist           
```

"""


from pathlib import Path
from data.data_file import Hand, Label, RecordFile
from typing import List
import numpy as np
import time
from dataclasses import dataclass

import argparse
import cv2
import mediapipe as mp



def convert_mediapipe_to_numpy(landmarks) -> np.ndarray:
    """Converts a mediapipe landmark point array to a numpy array"""
    return np.array([(e.x, e.y, e.z) for e in landmarks.landmark])

def record_landmarks(duration: float = 10.0) -> List[np.ndarray]:
    landmarks: List[np.ndarray] = []


    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0)

    end = time.time() + duration

    with mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as hands:
        while cap.isOpened() and time.time() < end:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks.append(convert_mediapipe_to_numpy(hand_landmarks))


            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

    return landmarks




def main_record():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=Label, choices=list(Label))
    parser.add_argument("hand", type=Hand, choices=list(Hand))
    parser.add_argument("-d","--duration", type=float)
    
    args = parser.parse_args()

    label: Label = args.label

    print(f"Recording label: {label}")
    landmarks = record_landmarks()

    
    path = Path("data").joinpath(label.name)
    path.mkdir(exist_ok=True)

    i = 0
    file = path.joinpath(f"run.{i}.json")
    while file.exists():
        i += 1
        file = path.joinpath(f"run.{i}.json")

    data = RecordFile(landmarks, args.hand)
    data.store(file)
    

if __name__ == "__main__":
    main_record()
