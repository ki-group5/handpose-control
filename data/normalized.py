"""
Starting with feature normalization of the detected hand skeleton and removing the positional data.

The goal is to have a position and orientation independent hand skeleton that then can be used
by a simple hand pose classifier. The orientation and positional data then can be used in a later step.

"""
from numpy.linalg import norm
from data.converter import convert_mediapipe_to_numpy
from dataclasses import dataclass
from typing import Union, Tuple

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
    handed_label: str  # "Left" or "Right" hand
    handed_score: float  # propability if left or right hand

    @classmethod
    def create(cls, landmarks, handed) -> "NormalizedData":
        data = convert_mediapipe_to_numpy(landmarks)
        dzero = data - data[HL.WRIST]

        # Calculate orienation
        u: np.ndarray = dzero[HL.INDEX_FINGER_MCP]
        v: np.ndarray = dzero[HL.PINKY_MCP]
        normal = np.cross(u, v)
        normal /= np.linalg.norm(normal)

        handed_label = list(handed.classification)[0].label
        handed_score = list(handed.classification)[0].score

        # print(handed.classification)

        if handed_label == "Left": # Invert normal when left hand...
            normal = -normal


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

        return cls(data, direction, normal, handed_label, handed_score)
