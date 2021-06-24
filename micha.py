"""
Starting with feature normalization of the detected hand skeleton and removing the positional data.

The goal is to have a position and orientation independent hand skeleton that then can be used
by a simple hand pose classifier. The orientation and positional data then can be used in a later step.

"""

from dataclasses import dataclass
from typing import Union

import cv2
import mediapipe as mp
import  numpy as np
from mediapipe.python.solutions.hands import HandLandmark as HL

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

Vec3 = Union[(float, float, float), np.ndarray]
Vec4 = Union[(float, float, float, float), np.ndarray]

def convert_to_numpy(landmarks) -> np.ndarray:
  return np.array([(e.x, e.y, e.z) for e in landmarks.landmark])


class Math:
  @staticmethod
  def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)

  def quat_rot_from_to(a: Vec3, b: Vec3) -> Vec4:
    xyz = np.cross(a, b)
    w = np.sqrt(np.linalg.norm(a)**2 * np.linalg.norm(b)**2) + a.dot(b)
    return np.array([*xyz, w])





@dataclass
class NPLandmarks:
  data: np.ndarray # Original data (nparray: 21x3)
  direction: np.ndarray # nparray: 20x3
  normal: np.ndarray # Orientation normal (vec3)

  @classmethod
  def create_from_landmarks(cls, landmarks) -> "NPLandmarks":
    data = convert_to_numpy(landmarks)
    dzero = data - data[HL.WRIST]

    # Calculate orienation
    u: np.ndarray = data[HL.INDEX_FINGER_MCP]
    v: np.ndarray = data[HL.PINKY_MCP]
    normal = np.cross(u,v)
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
    quat = quat_rot_from_to(normal, (1,0,0))






    return cls(data, direction, normal)




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

        l = NPLandmarks.create_from_landmarks(hand_landmarks)
        print(l)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
