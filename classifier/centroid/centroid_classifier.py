import os
import json
import numpy as np


class CentroidClassifier:
    def __init__(self, classifier_file=None):
        if classifier_file is None:
            classifier_file = os.path.dirname(os.path.abspath(__file__)) + '/params/01.json'
        deserialized_classifier = json.load(classifier_file)
        self.centroids = {
            label: np.array(centroid) for label, centroid in
            deserialized_classifier['classifier'].items()
        }
        self.cutoff = deserialized_classifier['cutoff']

    def classify(self, data):
        min_distance = self.cutoff
        best_label = None
        for label, centroid in self.centroids.items():
            distance = np.linalg.norm(centroid - data)
            if distance < min_distance:
                min_distance = distance
                best_label = label
        return best_label
