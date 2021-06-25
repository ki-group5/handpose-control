
from functools import cmp_to_key

from numpy.linalg import solve
from controller.state import Cmd
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np


class Confidence:
    def __init__(self, labels: List[str]) -> None:
        self.labels = labels
        self.values = np.zeros(len(labels))
        self.mvg_avg_factor = 4.0

    def update(self, dt: float, label: Optional[str] = None):
        delta = np.zeros(len(self.labels))

        if label:
            index = self.labels.index(label)
            delta[index] = 1.0

        mvgf = min(max(self.mvg_avg_factor * dt, 0.0), 1.0)
        self.values = (self.values * (1.0-mvgf)) + (delta * mvgf)
        # print(self.values)

    def max(self) -> Optional[Tuple[str, float]]:
        ind = np.argmax(self.values)
        return self.labels[ind], self.values[ind]

    def reset(self):
        self.values = np.zeros(len(self.labels))

    # def get(self) -> Optional[str]:
    #     ind = np.argmax(self.values)
    #     if self.values[ind] > 0.7:
    #         return self.labels[ind]
    #     return None


# class State:
#     def __init__(self, name: str, fallback: "State") -> None:
#         self.name = name
#         self.fallback_state = fallback
#         self.outgoing: Dict[str, Tuple(float, State)] = dict()
#         # self.on_visit: Optional[Callable] = None

#     def consume(self, conf: Confidence) -> "State":
#         label, value = conf.max()
#         if label in self.outgoing:
#             req_value, next_state = self.outgoing[label]
#             if req_value <= value:
#                 return next_state
#         else:
#             return self.fallback_state


@dataclass
class Cmd:
    label: str
    value: float


@dataclass
class Command:
    name: str
    chain: List[Cmd]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("Command called: ", self.name)


Commands: Dict[str, List[Cmd]] = {
    "stop": [Cmd("flat", 0.9), Cmd("fist", 0.9), Cmd("flat", 0.9)],
    "continue": [Cmd("index", 0.9), Cmd("fist", 0.9)]
}


class CommandState:
    def __init__(self, cmd: Command) -> None:
        self.cmd = cmd
        self.index = 0

    def reset(self):
        self.index = 0

    def consume(self, label: str, value: float) -> bool:
        element = self.cmd.chain[self.index]
        if element.label != label:
            return False
        
        # print(f"{value:.1f}, ", end=None)

        # print(element.value, value)
        if element.value <= value:
            self.index += 1
            if self.index == len(self.cmd.chain):
                # Trigger
                self.cmd()
                return True
        return False


class Commander:
    def __init__(self, commands: Dict[str, List[Cmd]], labels: List[str]) -> None:
        self.confidence = Confidence(labels)
        self.commands = [CommandState(Command(cname, clist))
                         for cname, clist in commands.items()]
        self.timestamp = time.time()

    def push(self, label: Optional[str]):
        t = time.time()
        dt = t - self.timestamp
        self.timestamp = t

        self.confidence.update(dt, label)
        mlabel, mconf = self.confidence.max()

        do_reset = False
        for c in self.commands:
            if c.consume(mlabel, mconf):
                do_reset= True
                break

        if do_reset:
            self.confidence.reset()
            for c in self.commands:
                c.reset()
