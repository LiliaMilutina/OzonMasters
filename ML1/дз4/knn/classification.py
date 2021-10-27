import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        classes = []
        for j in range(indices.shape[0]):
            votes = {}
            row = indices[j, :]
            for i in range(len(row)):
                if self._labels[row[i]] in votes:
                    if self._weights == 'distance':
                        votes[self._labels[row[i]]] += 1.0/(KNNClassifier.EPS+distances[j, i])
                    else:
                        votes[self._labels[row[i]]] += 1
                else:
                    if self._weights == 'distance':
                        votes[self._labels[row[i]]] = 1.0/(KNNClassifier.EPS+distances[j, i])
                    else:
                        votes[self._labels[row[i]]] = 1
            max_val = max(votes.values())
            temp = []
            for key, val in votes.items():
                if val == max_val:
                    temp.append(key)
            classes.append(min(temp))
        classes = np.asarray(classes)
        return classes

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedMixin:
    def __init__(self):
        self.batch_size = 100

    def kneighbors(self, X, return_distance=False):
        if not hasattr(self,  'batch_size'):
            self.batch_size = None

        batch_size = self.batch_size or X.shape[0]

        distances, indices = [], []

        for i_min in range(0, X.shape[0], batch_size):
            i_max = min(i_min + batch_size, X.shape[0])
            X_batch = X[i_min:i_max]

            indices_ = super().kneighbors(X_batch, return_distance=return_distance)
            if return_distance:
                distances_, indices_ = indices_
            else:
                distances_ = None

            indices.append(indices_)
            if distances_ is not None:
                distances.append(distances_)

        indices = np.vstack(indices)
        distances = np.vstack(distances) if distances else None

        result = (indices,)
        if return_distance:
            result = (distances,) + result
        return result if len(result) > 1 else result[0]

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


class BatchedKNNClassifier(BatchedMixin, KNNClassifier):
    def __init__(self, *args, **kwargs):
        KNNClassifier.__init__(self, *args, **kwargs)
        BatchedMixin.__init__(self)
