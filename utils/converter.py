import numpy as np


def convert_mediapipe_to_numpy(landmarks) -> np.ndarray:
    """Converts a mediapipe landmark point array to a numpy array"""
    return np.array([(e.x, e.y, e.z) for e in landmarks])

