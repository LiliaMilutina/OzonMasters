import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    indices = np.argpartition(ranks, top-1, axis=axis)[:, :top]
    values = np.take_along_axis(ranks, indices, axis=axis)
    ind_sort = np.argsort(values, axis=axis)
    res = np.take_along_axis(indices, ind_sort, axis=axis)
    return res


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        X_train = self._X
        indices = np.zeros((X.shape[0], self.n_neighbors)).astype(int)
        distances = np.zeros((X.shape[0], self.n_neighbors))
        ranks = self._metric_func(X, X_train)
        indices = get_best_ranks(ranks, self.n_neighbors, 1, return_distance)
        distances = np.take_along_axis(ranks, indices, axis=1)
        if return_distance is False:
            return indices
        else:
            return (distances, indices)
