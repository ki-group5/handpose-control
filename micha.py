"""
Starting with feature normalization of the detected hand skeleton and removing the positional data.

The goal is to have a position and orientation independent hand skeleton that then can be used
by a simple hand pose classifier. The orientation and positional data then can be used in a later step.

"""

from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from mediapipe.python.solutions.hands import HandLandmark as HL

from vec_math import Quat

Vec3 = Union[Tuple[float, float, float], np.ndarray]
Vec4 = Union[Tuple[float, float, float, float], np.ndarray]


def convert_mediapipe_to_numpy(landmarks) -> np.ndarray:
    return np.array([(e.x, e.y, e.z) for e in landmarks.landmark])


@dataclass
class NormalizedData:
    data: np.ndarray  # Original data (nparray: 21x3)
    direction: np.ndarray  # nparray: 20x3
    normal: np.ndarray  # Orientation normal (vec3)

    @classmethod
    def create_from_landmarks(cls, landmarks) -> "NPLandmarks":
        data = convert_mediapipe_to_numpy(landmarks)
        dzero = data - data[HL.WRIST]

        # Calculate orienation
        u: np.ndarray = dzero[HL.INDEX_FINGER_MCP]
        v: np.ndarray = dzero[HL.PINKY_MCP]
        normal = np.cross(u, v)
        normal /= np.linalg.norm(normal)

        # Calculate direction
        direction = np.array([
            # Thump
            dzero[1] - dzero[0],
            dzero[2] - dzero[1],
            dzero[3] - dzero[2],
            dzero[4] - dzero[3],
            # Index finger
            dzero[5] - dzero[0],
            dzero[6] - dzero[5],
            dzero[7] - dzero[6],
            dzero[8] - dzero[7],
            # Middle finger
            dzero[9] - dzero[0],
            dzero[10] - dzero[9],
            dzero[11] - dzero[10],
            dzero[12] - dzero[11],
            # Ring finger
            dzero[13] - dzero[0],
            dzero[14] - dzero[13],
            dzero[15] - dzero[14],
            dzero[16] - dzero[15],
            # Ring finger
            dzero[17] - dzero[0],
            dzero[18] - dzero[17],
            dzero[19] - dzero[18],
            dzero[20] - dzero[19],
        ])

        # Normalize the directions to unit vectors
        dlen = np.linalg.norm(direction, axis=1)
        dlen_sel = dlen != 0
        direction[dlen_sel] = (direction[dlen_sel].T / dlen).T

        # Rotate the directions so the normal is on the x-axis
        quat = Quat.create_from_to(normal, (1, 0, 0))
        direction = quat * direction

        return cls(data, direction, normal)


import math

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def example_normalization():
    """
    Just a main to show the normalization
    """
    import cv2
    import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
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

                    x = NormalizedData.create_from_landmarks(hand_landmarks)

                    # Draw normal into image
                    print(x.normal)

                    image_rows, image_cols, _ = image.shape

                    normal_pos0 = x.data[0]
                    normal_pos1 = normal_pos0 + x.normal * 0.1

                    p0 = _normalized_to_pixel_coordinates(
                        normal_pos0[0], normal_pos0[1], image_cols, image_rows)
                    p1 = _normalized_to_pixel_coordinates(
                        normal_pos1[0], normal_pos1[1], image_cols, image_rows)

                    if p0 and p1:
                        cv2.line(image, p0,                             p1,
                                (0, 100, 200),                             2)

                    # l = NPLandmarks.create_from_landmarks(hand_landmarks)
                    # print(l)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    example_normalization()
