
import numpy as np
from typing import Iterable, Tuple, Union
try:
    from csvread import csvread
except ImportError:
    from .csvread import csvread

def compress_ind(x, width):
    comp_arr = np.empty(width, dtype=np.float32)
    comp_arr[:], ind_width, i = np.NaN, int(np.round(x.size/width)), 0
    for ind in np.arange(x.size, step=ind_width, dtype=int):
        if i == comp_arr.size - 1:
            comp_arr[i] = np.sum(x[ind:]) / x[ind:].size
            return comp_arr[~np.isnan(comp_arr)], comp_arr[
                ~np.isnan(comp_arr)].size
        else:
            comp_arr[i] = np.sum(x[ind:ind + ind_width]) / x[ind:ind + ind_width].size
        i += 1
    return comp_arr[~np.isnan(comp_arr)], comp_arr[~np.isnan(comp_arr)].size


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
                return comp_arr[~np.isnan(comp_arr)], comp_arr[~np.isnan(comp_arr)].size
            else:
                continue
    return comp_arr[~np.isnan(comp_arr)], comp_arr[~np.isnan(comp_arr)].size

if __name__ == "__main__":
    from Tests.Helper_test import test_compress_ind, test_compress_width
    test_compress_width()
    test_compress_ind()
