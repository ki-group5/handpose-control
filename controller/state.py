
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from numpy.core.numeric import full


class Confidence:
    def __init__(self, labels: List[str]) -> None:
        self.labels = labels
        self.values = np.zeros(len(labels))
        self.mvg_avg_factor = 1.0

    def update(self, dt: float, label: Optional[str] = None):
        delta = np.zeros(len(self.labels))

        if label:
            index = self.labels.index(label)
            delta[index] = 1.0

        mvgf = min(max(self.mvg_avg_factor * dt, 0.0), 1.0)
        self.values = (self.values * (1.0-mvgf)) + (delta * mvgf)
        print(self.values)

    def get(self) -> Optional[str]:
        ind = np.argmax(self.values)
        if self.values[ind] > 0.7:
            return self.labels[ind]
        return None


class State:
    def __init__(self, label: Optional[str] = None) -> None:
        self.label = label
        self.timestamp = time.time()
        self.dt = 0

    def update_duration(self, t: float):
        self.dt = t - self.timestamp


class StateStream:
    def __init__(self, labels: List[str], max_length: int = 10) -> None:
        self.confidence = Confidence(labels)
        self.states: List[State] = [State("undefined")]
        self.timestamp = time.time()
        self.max_length = max_length

    @property
    def current(self) -> State:
        return self.states[-1]

    def _cleanup(self):
        if len(self.states) > self.max_length:
            self.states = self.states[-self.max_length:]

    def clear(self):
        self.states = [self.states[-1]]

    def push(self, label: Optional[str] = None) -> Optional[Tuple[str, str]]:
        t = time.time()
        dt = t - self.timestamp
        self.timestamp = t

        self.confidence.update(dt, label)
        conf_label = self.confidence.get()

        self.current.update_duration(t)
        old_label = self.current.label
        if conf_label and old_label != conf_label:
            self.states.append(State(conf_label))
            self._cleanup()
            return (old_label, conf_label)

        return None




@dataclass
class Cmd:
    label: str
    min_time: float = 0.2
    max_time: float = 20.0


Commands: Dict[str, List[Cmd]] = {
    "stop": [Cmd("flat", 0.8, 2.0), Cmd("fist", 0.2, 1.0), Cmd("flat", 0.8, 20.0)],
    "continue": [Cmd("index", 0.3, 20.0), Cmd("fist", 1.0, 2.0)]
}


class Commander:
    def __init__(self, commands: Dict[str, List[Cmd]]) -> None:
        self.commands = commands
        self.command_state_change: Dict[str, List[Tuple[str, str]]] = dict()
        for cname, clist in commands.items():
            self.command_state_change[cname] = list(zip(clist[:-1], clist[1:]))

        self.max_length = 10
        self.history: List[Tuple[str, str]] = []

    def add_state_change(self, change: Tuple[str, str]) -> Optional[str]:
        self.history.append(change)
        self.history = self.history[-self.max_length:]
        self._detect_command()

    def _detect_command(self) -> Optional[str]:
        for cname, ctupl in self.command_state_change.items():
            if self.history[-len(ctupl):] == ctupl:
                return cname
        return None


# def detect_cmd(label: str, commands: List[Cmd], sstream: StateStream, skip_time: float = 0.3) -> bool:
#     states = reversed(sstream.states)
#     for c in reversed(commands):
#         time_skip = skip_time
#         time_total = 0.0
#         fullfilled = False

#         while time_skip > 0 and not fullfilled:
#             try:
#                 s = next(states)
#                 # print(s.label, "==", c.label)
#                 # print(time_total, ", ", time_skip, ", dt:", s.dt)
#                 if s.label == c.label:
#                     time_total += s.dt
#                     if time_total >= c.min_time:
#                         fullfilled = True
#                 else:
#                     time_skip -= s.dt
#             except StopIteration:
#                 return False
#         if time_skip < 0:
#             return False
#         if time_total < c.min_time or c.max_time < time_total:
#             return False
#     return True


# def detect_commands(commands: Dict[str, List[Cmd]], sstream: StateStream, skip_time: float = 0.3) -> Optional[str]:
#     for cname, clist in commands.items():
#         if detect_cmd(cname, clist, sstream, skip_time):
#             return cname
#     return None


def demo():
    labels = ["undefined", "fist", "index"]
    stream = StateStream(labels, 10)
    commander = Commander(Commands)

    def wow(label, dt: float = 0.1):
        def bar():
            time.sleep(dt)

            dstate = stream.push(label)
            if dstate:
                print(f"State changed: {stream.states[-2].label} -> {stream.states[-1].label}")
                cmd = commander.add_state_change(Commands, stream)
                if cmd:
                    stream.clear()
                    print(cmd)
        bar()
        bar()
        bar()
        bar()

    wow("undefined")
    wow("undefined")
    wow("undefined")
    wow("undefined")
    wow("undefined")
    wow("fist")
    wow("fist")
    wow("fist")
    wow("fist")
    wow("fist")
    wow("fist")
    wow("index")
    wow("index")
    wow("index")
    wow("index")
    wow("index")
    wow("index")
    wow("index")
    wow("fist")
    wow("fist")
    wow("fist")
    wow("fist")
    wow("fist")


if __name__ == "__main__":
    demo()
