import pickle


class KMeansClassifier:
    def __init__(self, classifier_file='kmeans_classifier.pickle'):
        with open(classifier_file, 'rb') as f:
            serialized_classifier = f.read()
        deserialized_classifier = pickle.loads(serialized_classifier)
        self.classifier = deserialized_classifier['classifier']
        self.cutoff = deserialized_classifier['cutoff']

    def classify(self, data):
        """Find closest cluster to data point. Return None if distance is larger than cutoff."""
        cluster_distance_space = self.classifier.transform([data])[0]
        print(self.classifier.labels_)
        min_index = cluster_distance_space.argmin()
        min_distance = cluster_distance_space[min_index]
        if min_distance < self.cutoff:
            return min_index
