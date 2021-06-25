import os
from pathlib import Path
from typing import List
import numpy as np
import json
from enum import Enum
from dataclasses import dataclass


class Label(Enum):
    Undefined = "undefined"
    Flat = "flat"
    fist = "fist"
    Index = "index"
    Stop = "stop"
    Ok = "ok"
    Arrow = "arrow"
    ThumbUp = "thumbup"
    # ThumbDown = "thumbdown"

    def __str__(self) -> str:
        return self.value


class Hand(Enum):
    Left = "left"
    Right = "right"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get(cls, name: str):
        if name == "left":
            return cls.Left
        elif name == "right":
            return cls.Right
        raise ValueError


@dataclass
class RecordFile:
    landmarks: List[np.ndarray]
    hand: Hand

    def store(self, path: Path):
        obj = {
            "landmark_frames": [x.tolist() for x in self.landmarks],
            "hand": self.hand.value
        }
        with path.open('wt') as fp:
            json.dump(obj, fp)

    @classmethod
    def load(cls, path) -> "RecordFile":
        with open(path, mode="rt") as fp:
            obj = json.load(fp)
        return cls(
            [np.asarray(x) for x in obj["landmark_frames"]],
            Hand.get(obj["hand"])
        )
