import json
import os
import numpy as np


def load_data():
    dataset = {}
    datadir = f'{os.path.dirname(os.path.abspath(__file__))}/data/'
    labels = os.listdir(datadir)
    datafolders = [datadir + label for label in labels]
    labels = filter(lambda f: len(os.listdir(datadir + f)) > 0, datafolders)  # filter empty folders
    for label in labels:
        datafolder = datadir + label
        for datafile in os.listdir(datafolder):
            if os.path.splitext(datafile)[1] != '.json':
                continue
            with open(datafile) as f:
                dataset[label] = np.array(json.load(f)['landmark_frames'])
    return dataset


if __name__ == '__main__':
    print(load_data())
