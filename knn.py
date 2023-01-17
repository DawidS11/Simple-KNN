import numpy as np
import itertools


def euclidean_distance(a, b):
    dist = float(sum([(a[i] - b[i])**2 for i in range(len(a))]))
    dist = np.sqrt(dist)

    return dist


def knn(x_train, y_train, x_test, k):
    y_test = [-1] * len(x_test)      # predictions are >= 0

    for idx, p_test in enumerate(x_test):
        # calculate euclidean distance to each point in x_train and classify  
        neighbors = {}
        for idx2, p_train in enumerate(x_train):
            neighbors[idx2] = euclidean_distance(p_train, p_test)
        sorted_neighbors = dict(sorted(neighbors.items(), key=lambda item: item[1]))

        closest_neighbors = (dict(itertools.islice(sorted_neighbors.items(), k)))
        closest_neighbors = [y_train[i] for i in list(closest_neighbors.keys())]

        y_test[idx] = max(set(closest_neighbors), key = closest_neighbors.count)

    return y_test