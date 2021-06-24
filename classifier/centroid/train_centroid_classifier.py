import json
import os
import numpy as np

from ...load_data import load_data


def train_centroid_classifier(model_filename='01.json'):
    data = load_data()
    labels = data.keys()

    centroids = {}
    for label in labels:
        _data = data[label]
        num_datapoints = _data.shape[0]
        centroid = _data.sum(axis=0) / num_datapoints
        centroids[label] = centroid

    # find cutoff
    cutoff = 0
    for label, centroid in centroids.items():
        max_distance = np.linalg.norm(data[label] - centroid, axis=(1, 2)).max()
        cutoff = max(cutoff, max_distance)

    classifier_data = {
        'classifier': centroids,
        'cutoff': cutoff
    }

    model_filename = f'{os.path.dirname(os.path.abspath(__file__))}/params/{model_filename}'
    with open(model_filename, 'w') as f:
        json.dump(classifier_data, f)
