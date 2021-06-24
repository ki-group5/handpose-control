import json
import numpy as np
import os


def train_centroid_classifier(datafile, model_filename='01.json'):
    data = json.load(datafile)

    shape = data[0].data.shape
    unique_labels = set(data.labels)
    centroids = {label: [np.zeros(shape), 0] for label in unique_labels}

    for d in data:
        c = centroids[d.label]
        c[0] += d.data
        c[1] += 1

    for label, centroid in centroids:
        centroid = centroid[0] / centroid[1]
        centroids[label] = centroid

    model_filename = f'{os.path.dirname(os.path.abspath(__file__))}/params/{model_filename}'
    with open(model_filename, 'w') as f:
        json.dump(centroids, f)
