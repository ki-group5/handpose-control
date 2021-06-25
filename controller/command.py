
from data.data_file import Label
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


DEFAULT_COMMANDS: Dict[str, List[Cmd]] = {
    "stop": [Cmd("flat", 0.8), Cmd("fist", 0.8), Cmd("flat", 0.8)],
    "continue": [Cmd("index", 0.8), Cmd("fist", 0.8)]
}


class CommandState:
    def __init__(self, cmd: Command) -> None:
        self.cmd = cmd
        self.index = 0

    def reset(self):
        self.index = 0

    def notify_change(self, label: str) -> bool:
        element = self.cmd.chain[self.index]
        # print(self.index, element.label)
        if element.label == label:
            self.index += 1
            if self.index == len(self.cmd.chain):
                self.index = 0
                return True
        return False

class Commander:
    def __init__(self, labels: List[str], commands: Dict[str, List[Cmd]] = None) -> None:
        self.confidence = Confidence(labels)
        self.current_label: str = Label.Undefined.value

        if commands is None:
            commands = DEFAULT_COMMANDS

        self.commands = [CommandState(Command(cname, clist))
                         for cname, clist in commands.items()]
        self.timestamp = time.time()

    def push(self, label: Optional[str]) -> Optional[str]:
        # print(label)
        t = time.time()
        dt = t - self.timestamp
        self.timestamp = t

        self.confidence.update(dt, label)
        mlabel, mconf = self.confidence.max()

        
        cmd: Optional[str] = None
        if label is not None:
            if mlabel != self.current_label:
                # State changed!!!
                # print("State changed", mlabel, mconf)
                self.current_label = mlabel
                for c in self.commands:
                    if c.notify_change(mlabel):
                        cmd = c.cmd.name
                        break

            if cmd:
                for c in self.commands:
                    c.reset()
        
        return cmd

