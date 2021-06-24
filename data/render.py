

from data.normalized import NormalizedData
import math
from typing import Optional, Tuple

import cv2


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
