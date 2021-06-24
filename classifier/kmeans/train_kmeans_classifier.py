from sklearn.cluster import KMeans
import numpy as np
import pickle
import json
from collections import Counter

RANDOM_SEED = 0
# VERBOSE = True
VERBOSE = False


def train_kmeans_classifier(datafile, model_filename='kmeans_classifier.pickle'):
    print('Loading data...')
    with open(datafile) as f:
        data = json.loads(f.read())

    labels = data['labels']
    unique_labels = set(labels)
    data = data['data']
    num_poses = len(unique_labels)

    print('Fitting data...')
    kmeans = KMeans(n_clusters=num_poses, verbose=VERBOSE, random_state=RANDOM_SEED)
    clustered_data = kmeans.fit_predict(data)

    print('Checking clusters...')
    # data has been clustered, but we don't know yet which label they correspond with
    # to find out which cluster index corresponds with which label, we search for maximum
    cl_check = Counter(zip(labels, clustered_data))
    cl_detections = {label: {} for label in labels}
    for (label, cluster), count in cl_check.most_common():
        cl_detections[label][cluster] = count

    label_mapping = {}
    print('Label -> Cluster Index (Prob%), ...')
    for label, counts in cl_detections.items():
        sum_counts = sum(counts.values())
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)  # sort descending by count
        label_mapping[sorted_counts[0][0]] = label  # store mapping from cluster index to label
        s = f'{label} -> ' + ', '.join([f' {cl} ({100 * count / sum_counts}%)' for cl, count in sorted_counts])
        print(s)

    cluster_distances = kmeans.transform(data)
    cutoff = cluster_distances.min(axis=1).max()  # Use max distance of data point to its cluster as cutoff value

    serialized_classifier = pickle.dumps({
        'classifier': kmeans,
        'cutoff': cutoff,
        'label_mapping': label_mapping
    })
    with open(model_filename, 'wb') as f:
        f.write(serialized_classifier)


if __name__ == '__main__':
    train_kmeans_classifier('../../data/test_data.json')
