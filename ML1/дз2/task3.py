def get_max_after_zero(x):
    import numpy as np
    ind_zero = np.where(x == 0)[0]
    ind_el = np.where(ind_zero <= x.shape[0]-2)[0]
    res = x[ind_zero[ind_el]+1]
    if len(res) == 0:
        return None
    else:
        return max(res)
