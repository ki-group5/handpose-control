import json
import os
import numpy as np


def load_data():
    dataset = {}
    datadir = f'{os.path.dirname(os.path.abspath(__file__))}/data/'
    labels = filter(lambda f: os.path.isdir(datadir + f), os.listdir(datadir))
    for label in labels:
        datafolder = datadir + label + '/'
        for datafile in os.listdir(datafolder):
            if os.path.splitext(datafile)[1] != '.json':
                continue
            with open(datafolder + datafile) as f:
                dataset[label] = np.array(json.load(f)['landmark_frames'])
    return dataset


if __name__ == '__main__':
    print(load_data())
