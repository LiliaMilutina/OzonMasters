def get_nonzero_diag_product(X):
    import numpy as np
    diag_el = np.diagonal(X)
    ind_non_zero = np.where(diag_el != 0)
    non_zero = diag_el[ind_non_zero]
    if len(non_zero) == 0:
        return None
    else:
        return np.prod(non_zero)
