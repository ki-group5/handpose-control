

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List
from data.data_file import Label, RecordFile, Hand
from pathlib import Path
import numpy as np


@dataclass
class LabelledEntry:
    label: Label
    landmarks: np.ndarray
    hand: Hand



class DataLoader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def iter_load_all(self) -> Iterator[LabelledEntry]:
        for p in self.path.iterdir():
            if p.is_dir():
                try:
                    label = Label[p.name]
                except KeyError:
                    continue
                for f in p.iterdir():
                    if f.name.startswith("run.") and f.name.endswith(".json"):
                        file = RecordFile.load(f.resolve())
                        for lm in file.landmarks:
                            yield LabelledEntry(label, lm, file.hand)

    def load_all(self) -> List[LabelledEntry]:
        return list(self.iter_load_all())


if __name__ == "__main__":

    # Test

    loader = DataLoader(Path("data/"))
    
    wow = list(loader.load_all())

    print(wow)
