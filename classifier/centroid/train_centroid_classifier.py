import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import DataLoader
from utils.distance_normalization import NormalizedData


def train_centroid_classifier(model_filename='01.json'):
    loader = DataLoader(Path("../../data/"))
    data = loader.load_all()
    unique_labels = set(entry.label for entry in data)
    centroids = {label: [] for label in unique_labels}

    for entry in data:
        normalized_data = NormalizedData.create(entry.landmarks, entry.hand.name)
        centroids[entry.label].append(normalized_data.distance)

    data = {}
    for label in unique_labels:
        _data = np.array(centroids[label])
        data[label] = _data
        num_datapoints = len(_data)
        centroid = _data.sum(axis=0) / num_datapoints
        centroids[label] = centroid

    # find cutoff
    cutoff = 0
    _distances = []
    for label, centroid in centroids.items():
        print(data[label])
        print(centroid)
        print(data[label].shape)
        print(centroid.shape)
        print(data[label] - centroid)
        print((data[label] - centroid).shape)
        distances = np.linalg.norm(data[label] - centroid, axis=1)
        print(distances.shape)
        _distances.append(distances)
        max_distance = distances.max()
        cutoff = max(cutoff, max_distance)

    classifier_data = {
        'classifier': {str(label): centroid.tolist() for label, centroid in centroids.items()},
        'cutoff': cutoff
    }

    model_filename = f'{os.path.dirname(os.path.abspath(__file__))}/params/{model_filename}'
    Path(model_filename).parent.mkdir(exist_ok=True)
    with open(model_filename, 'w') as f:
        json.dump(classifier_data, f)

    import matplotlib.pyplot as plt
    distances = np.concatenate(_distances)
    quantiles = pd.Series(distances)
    quantiles.plot(kind='hist', bins=20)
    print(f'Consider adjusting cutoff in {model_filename} as necessary according to distance histogram!')
    plt.show()


if __name__ == '__main__':
    train_centroid_classifier('02.json')
