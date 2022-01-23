import numpy as np
from typing import Iterable, Tuple, Union

try:
    from csvread import csvread
except ImportError:
    from .csvread import csvread


def compress_ind(x: np.ndarray, width: int):
    """
    Compress array to a given size (width).
    :param x:
        Array
    :param width:
        Total size of the new array.
    :return:
        Compressed array.
    """
    comp_arr = np.empty(width, dtype=np.float32)
    comp_arr[:], ind_width, i = np.NaN, int(np.round(x.size / width)), 0
    # for all new indexes.
    for ind in np.arange(x.size, step=ind_width, dtype=int):
        # if the end of the array is reached add all last values to account
        # for values that don't fit in the resulting array.
        if i == comp_arr.size - 1:
            comp_arr[i] = np.sum(x[ind:]) / x[ind:].size
            return comp_arr[~np.isnan(comp_arr)], comp_arr[
                ~np.isnan(comp_arr)].size
        # Add all values between the index and the new index width to the
        # new array.
        else:
            comp_arr[i] = np.sum(x[ind:ind + ind_width]) / x[
                                                           ind:ind + ind_width].size
        i += 1
    return comp_arr[~np.isnan(comp_arr)], comp_arr[~np.isnan(comp_arr)].size


def compress_width(x: np.ndarray, width: Union[int, float]):
    """
    Compress array with given width per index.
    :param x:
        Array
    :param width:
        Width of 1 index
    :return:
        Compressed array.
    """
    comp_arr = np.empty(int(np.round(abs(max(x) - min(x)) / width)),
                        dtype=np.float32)
    comp_arr[:], i, k = np.NaN, 0, 0
    # For all indexes, ind.
    for ind in range(comp_arr.size):
        ind = k
        # For all indexes > ind
        for j in range(ind, x.size):
            # If the absolute difference is larger than the given width all
            # indexes between ind and j will be averaged and added to comp_arr
            if abs(x[ind] - x[j]) >= width:
                comp_arr[i] = np.sum(x[ind:j]) / x[ind:j].size
                i += 1
                k = j
                break
            # If the last index is reached and the width requirement is not met
            # add these to the comp arr
            elif x[j] == x[-1]:
                comp_arr[i] = np.sum(x[ind:]) / x[ind:].size
                return comp_arr[~np.isnan(comp_arr)], comp_arr[
                    ~np.isnan(comp_arr)].size
            else:
                continue
    return comp_arr[~np.isnan(comp_arr)], comp_arr[~np.isnan(comp_arr)].size


if __name__ == "__main__":
    from Tests.Helper_test import test_compress_ind, test_compress_width

    test_compress_width()
    test_compress_ind()
