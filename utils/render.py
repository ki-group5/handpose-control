

from utils.normalized import NormalizedData
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

import cv2
from mediapipe.python.solutions.drawing_utils import DrawingSpec


class PoseRender:

    @classmethod
    def normalized_to_pixel_coordinates(cls,
                                        normalized_x: float, normalized_y: float, image_width: int,
                                        image_height: int) -> Optional[Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates.

        From: mediapipe/python/solutions/drawing_utils.py
        """
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

    @classmethod
    def draw_normal(cls, data: NormalizedData, image):
        image_rows, image_cols, _ = image.shape

        normal_pos0 = data.data[0]
        normal_pos1 = normal_pos0 + data.normal * 0.1

        p0 = cls.normalized_to_pixel_coordinates(
            normal_pos0[0], normal_pos0[1], image_cols, image_rows)
        p1 = cls.normalized_to_pixel_coordinates(
            normal_pos1[0], normal_pos1[1], image_cols, image_rows)

        if p0 and p1:
            cv2.line(image, p0, p1, (0, 100, 200),  2)

    def render_landmarks(
            image: np.ndarray,
            landmark_list: np.ndarray,
            connections: Optional[List[Tuple[int, int]]] = None,
            landmark_drawing_spec: DrawingSpec = DrawingSpec(
                color=(0, 0, 255)),
            connection_drawing_spec: DrawingSpec = DrawingSpec()):
        if len(landmark_list) == 0:
            return
        if image.shape[2] != 3:
            raise ValueError(
                'Input image must contain three channel rgb data.')
        image_rows, image_cols, _ = image.shape
        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list):
            # if ((landmark.HasField('visibility') and
            #      landmark.visibility < VISIBILITY_THRESHOLD) or
            #     (landmark.HasField('presence') and
            #      landmark.presence < 0.5)):
            #   continue
            landmark_px = PoseRender.normalized_to_pixel_coordinates(landmark[0], landmark[1],
                                                                     image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
        if connections:
            num_landmarks = len(landmark_list)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                    cv2.line(image, idx_to_coordinates[start_idx],
                            idx_to_coordinates[end_idx], connection_drawing_spec.color,
                            connection_drawing_spec.thickness)
        # Draws landmark points after finishing the connection lines, which is
        # aesthetically better.
        for landmark_px in idx_to_coordinates.values():
            cv2.circle(image, landmark_px, landmark_drawing_spec.circle_radius,
                       landmark_drawing_spec.color, landmark_drawing_spec.thickness)
