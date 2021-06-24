from typing import Union, Tuple, List

import numpy as np

Vec3f = Union[Tuple[float, float, float], np.ndarray]
Vec4f = Union[Tuple[float, float, float, float], np.ndarray]


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


class Quat:
    """A quaternion that can rotate a single vector or a list of vectors"""
    def __init__(self, xyzw: Vec4f):
        self.xyzw = np.asarray(xyzw)

    @classmethod
    def create_from_to(cls, vec_from: Vec3f, vec_to: Vec3f) -> "Quat":
        a = np.asarray(vec_from)
        b = np.asarray(vec_to)
        xyz = np.cross(a, b)
        w = np.sqrt(np.linalg.norm(a)**2 * np.linalg.norm(b)**2) + a.dot(b)
        xyzw = np.array([*xyz, w])
        return cls(xyzw / np.linalg.norm(xyzw))

    def apply_to(self, other: Union[Vec3f, List[Vec3f], np.ndarray]):
        o = np.asarray(other)
        if o.shape == (3,):
            w = self.xyzw[3]
            b = self.xyzw[:3]
            b2 = np.dot(b, b)
            return o * (w * w - b2) + (b * (np.dot(o, b) * 2)) + (np.cross(b, o) * (2 * w))
        elif o.ndim == 2 and o.shape[1] == 3:
            w = self.xyzw[3]
            b = self.xyzw[:3]
            b2 = np.dot(b, b)
            return (np.outer(np.dot(o, b) * 2, b)
                    + (np.cross(b, o) * (2 * w))
                    + o * (w * w - b2)
                    )

        raise ValueError(f"Unkown shape: {o.shape}")

    def __mul__(self, other: Union[Vec3f, List[Vec3f], np.ndarray]):
        return self.apply_to(other)

    def to_vec(self) -> Vec4f:
        return self.xyzw

    def __str__(self):
        return str(self.to_vec())

