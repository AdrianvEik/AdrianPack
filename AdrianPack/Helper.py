
import numpy as np
from typing import Iterable, Tuple, Union
try:
    from csvread import csvread
except ImportError:
    from .csvread import csvread

def compress_ind(x, width):
    return None


def compress_width(x, width):
    comp_arr = np.empty(int(np.round(abs(max(x)-min(x))/width)), dtype=np.float32)
    comp_arr[:], i, k = np.NaN, 0, 0
    for ind in range(comp_arr.size):
        ind = k
        for j in range(ind, x.size):
            if abs(x[ind] - x[j]) >= width:
                comp_arr[i] = np.sum(x[ind:j])/x[ind:j].size
                i += 1
                k = j
                break
            elif x[j] == x[-1]:
                comp_arr[i] = np.sum(x[ind:])/x[ind:].size
                return comp_arr[~np.isnan(comp_arr)]
            else:
                continue

    return comp_arr[~np.isnan(comp_arr)]

if __name__ == "__main__":
    from Tests.Helper_test import test_compress_width
    test_compress_width()
