import numpy as np


def get_best_indices(ranks: np.ndarray, top: int, axis: int = 1) -> np.ndarray:
    indices = np.argpartition(ranks, -top, axis=axis)
    indices_new = indices.take(indices=range(indices.shape[axis]-top, indices.shape[axis]), axis=axis)
    values = np.take_along_axis(ranks, indices_new, axis=axis)
    ind_sort = np.argsort(-values, axis=axis)
    res = np.take_along_axis(indices_new, ind_sort, axis=axis)
    return res


if __name__ == "__main__":
    with open('input.bin', 'rb') as f_data:
        ranks = np.load(f_data)
    indices = get_best_indices(ranks, 5)
    with open('output.bin', 'wb') as f_data:
        np.save(f_data, indices)
