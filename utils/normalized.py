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
    direction: np.ndarray  # nparray: 20x3
    normal: np.ndarray  # Orientation normal (vec3)
    hand: str  # "Left" or "Right" hand
    rotation: Optional[np.ndarray] = None

    @classmethod
    def create_from_mediapipe(cls, landmarks, hand: str) -> "NormalizedData":
        data = convert_mediapipe_to_numpy(landmarks)
        return cls.create(data, hand)

    @classmethod
    def create(cls, data, hand: str) -> "NormalizedData":
        dzero = data - data[HL.WRIST]

        # Calculate orienation
        u: np.ndarray = dzero[HL.INDEX_FINGER_MCP]
        v: np.ndarray = dzero[HL.PINKY_MCP]
        normal = np.cross(u, v)
        normal /= np.linalg.norm(normal)

        # print(hand.classification)

        if hand == "Left":  # Invert normal when left hand...
            normal = -normal

        # Calculate direction
        direction = np.array([
            # Thumb
            dzero[1],
            dzero[2] - dzero[1],
            dzero[3] - dzero[2],
            dzero[4] - dzero[3],
            # Index finger
            dzero[5],
            dzero[6] - dzero[5],
            dzero[7] - dzero[6],
            dzero[8] - dzero[7],
            # Middle finger
            dzero[9],
            dzero[10] - dzero[9],
            dzero[11] - dzero[10],
            dzero[12] - dzero[11],
            # Ring finger
            dzero[13],
            dzero[14] - dzero[13],
            dzero[15] - dzero[14],
            dzero[16] - dzero[15],
            # Pinky finger
            dzero[17],
            dzero[18] - dzero[17],
            dzero[19] - dzero[18],
            dzero[20] - dzero[19],
        ])

        # Normalize the directions to unit vectors
        dlen = np.linalg.norm(direction, axis=1)
        dlen_sel = dlen != 0
        direction[dlen_sel] = (direction[dlen_sel].T / dlen[dlen_sel]).T

        # # Rotate the directions so the normal is on the x-axis
        # x = np.array([u, v, normal])
        # rotation = np.linalg.inv(np.eye(3))
        # direction = rotation.dot(direction.T).T

        # quat = Quat.create_from_to(normal, (1, 0, 0))
        # direction = quat * direction

        return cls(data, direction, normal, hand, None)

    def reconstruct(self) -> np.ndarray:
        # d = self.direction * 0.05
        d = self.direction * 0.05

        data = np.array([
            (0, 0, 0), d[0], d[0]+d[1], d[0]+d[1]+d[2], d[0]+d[1]+d[2]+d[3],
            d[4], d[4]+d[5], d[4]+d[5]+d[6], d[4]+d[5]+d[6]+d[7],
            d[8], d[8]+d[9], d[8]+d[9]+d[10], d[8]+d[9]+d[10]+d[11],
            d[12], d[12]+d[13], d[12]+d[13]+d[14], d[12]+d[13]+d[14]+d[15],
            d[16], d[16]+d[17], d[16]+d[17]+d[18], d[16]+d[17]+d[18]+d[19]
        ])
        # quat = Quat.create_from_to(self.normal, (1,0,0))
        # data = quat * data

        data += self.data[0]

        return data
