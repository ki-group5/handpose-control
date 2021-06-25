"""
Starting with feature normalization of the detected hand skeleton and removing the positional data.

The goal is to have a position and orientation independent hand skeleton that then can be used
by a simple hand pose classifier. The orientation and positional data then can be used in a later step.

"""
from numpy.linalg import norm
from utils.converter import convert_mediapipe_to_numpy
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np
from mediapipe.python.solutions.hands import HandLandmark as HL

from vec_math import Quat

Vec3 = Union[Tuple[float, float, float], np.ndarray]
Vec4 = Union[Tuple[float, float, float, float], np.ndarray]


@dataclass
class NormalizedData:
    data: np.ndarray  # Original data (nparray: 21x3)
    distance: np.ndarray  # nparray: 20x3
    hand: str  # "Left" or "Right" hand

    @classmethod
    def create_from_mediapipe(cls, landmarks, hand: str) -> "NormalizedData":
        data = convert_mediapipe_to_numpy(landmarks)
        return cls.create(data, hand)

    @classmethod
    def create(cls, data, hand: str) -> "NormalizedData":
        middle_of_hand = (data[0] + data[5] + data[9] + data[13] + data[17]) * .2
        thumb_direction = data[4] - data[3]
        thumb_direction /= norm(thumb_direction)
        up = np.array([0, 1, 0])
        hand_base_direction_1 = data[5] - data[17]
        hand_base_direction_1 /= norm(hand_base_direction_1)
        hand_base_direction_2 = data[5] - data[0]
        hand_base_direction_2 /= norm(hand_base_direction_2)

        index_base_direction = data[5] - data[0]
        middle_base_direction = data[9] - data[0]
        ring_base_direction = data[13] - data[0]
        little_base_direction = data[17] - data[0]

        distance = np.array([
            # thumb - index dist
            norm(data[4] - data[8]) ** 2,
            # index - middle distance
            norm(data[8] - data[12]),
            # middle - ring distance
            norm(data[12] - data[16]),
            # ring - little distance
            norm(data[16] - data[20]),
            # finger tip to hand palm distance
            norm(data[4] - middle_of_hand) ** 2,
            norm(data[8] - middle_of_hand) ** 2,
            norm(data[12] - middle_of_hand) ** 2,
            norm(data[16] - middle_of_hand) ** 2,
            norm(data[20] - middle_of_hand) ** 2,
            # upwardness of thumb upper segment (might be dumb)
            # norm(thumb_direction - up),
            # hand orientation
            # norm(hand_base_direction_1 - up),
            # norm(hand_base_direction_2 - up)
        ])

        return cls(data, distance, hand)
