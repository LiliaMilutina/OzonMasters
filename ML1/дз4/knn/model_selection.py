from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import KNNClassifier, BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))
    d = dict()
    k_max = max(k_list)
    clf = BatchedKNNClassifier(n_neighbors=k_max, **kwargs)
    scores = []
    for train_index, test_index in cv.split(X):
        sc = []
        clf.fit(X[train_index], y[train_index])
        distanses, indices = clf.kneighbors(X[test_index], return_distance=True)
        for k in k_list:
            distanses_new = distanses[:, :k]
            indices_new = indices[:, :k]
            y_pred = clf._predict_precomputed(indices_new, distanses_new)
            sc.append(accuracy_score(y_pred, y[test_index]))
        scores.append(sc)
    for i in range(len(k_list)):
        temp = []
        for el in scores:
            temp.append(el[i])
        d[k_list[i]] = np.asarray(temp)
    return d
