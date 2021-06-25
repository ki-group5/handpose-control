import sys
sys.path[0] += '\\..'
sys.path[0] += '\\..'

import json
import os
from pathlib import Path

import numpy as np

from data.loader import DataLoader
from utils.normalized import NormalizedData


def train_centroid_classifier(model_filename='01.json'):
    loader = DataLoader(Path("./data/"))
    data = loader.load_all()
    unique_labels = set(entry.label for entry in data)
    centroids = {label: [] for label in unique_labels}

    for entry in data:
        normalized_data = NormalizedData.create(entry.landmarks, entry.hand.name)
        #print(entry.hand.name)
        #print(entry.label)
        #print(normalized_data)
        # TODO: use normalized_data.normal ?
        centroids[entry.label].append(normalized_data.direction)

    data = {}
    for label in unique_labels:
        _data = np.array(centroids[label])
        data[label] = _data
        num_datapoints = len(_data)
        centroid = _data.sum(axis=0) / num_datapoints
        centroids[label] = centroid

    # find cutoff
    cutoff = 0
    for label, centroid in centroids.items():
        max_distance = np.linalg.norm(data[label] - centroid, axis=(1, 2)).max()
        cutoff = max(cutoff, max_distance)

    classifier_data = {
        'classifier': {str(label): centroid.tolist() for label, centroid in centroids.items()},
        'cutoff': cutoff
    }

    model_filename = f'{os.path.dirname(os.path.abspath(__file__))}/params/{model_filename}'
    with open(model_filename, 'w') as f:
        json.dump(classifier_data, f)


if __name__ == '__main__':
    train_centroid_classifier()
