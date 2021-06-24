from sklearn.cluster import KMeans
import numpy as np
import pickle
import json

RANDOM_SEED = 0
VERBOSE = True


def train_kmeans_classifier(datafile, model_filename):
    print('Loading data...')
    with open(datafile) as f:
        data = json.loads(f.read(datafile))

    labels = set(data['labels'])
    num_poses = len(labels)

    # TODO: Label clusters
    print('Fitting data...')
    kmeans = KMeans(n_clusters=num_poses, verbose=VERBOSE, random_state=RANDOM_SEED).fit(data)
    cutoff = kmeans.transform(data).min(axis=1).max()  # Use max distance of data point to its cluster as cutoff value
    print('Done!')

    serialized_classifier = pickle.dumps({
        'classifier': kmeans,
        'cutoff': cutoff,
        'labels': labels
    })
    with open(model_filename, 'wb') as f:
        f.write(serialized_classifier)
