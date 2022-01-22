import numpy as np
import numpy.testing as test
import time
from Helper import compress_width, compress_ind

def test_compress_width():
    arr, t = np.array(range(1, 30), dtype=np.float32), time.time_ns()
    test.assert_array_almost_equal(compress_width(arr, 5)[0],
                                   np.array([3, 8, 13, 18, 23, 27.5], dtype=np.float32),
                                   decimal=7)
    print("Test 1 Passed in %s ns" % (time.time_ns() - t))
    arr, t = np.array([0.21, 0.29, 0.36, 0.57, 0.59, 0.65, 0.73, 0.85, 0.88, 0.93], dtype=np.float32), time.time_ns()
    test.assert_array_almost_equal(compress_width(arr, 0.2)[0],
                                   np.array([0.86/3, 2.54/4, 2.66/3], dtype=np.float32),
                                   decimal=7)
    print("Test 2 Passed in %s ns" % (time.time_ns() - t))
    t = time.time_ns()
    test.assert_array_almost_equal(compress_width(np.flip(arr), 0.2)[0],
                                   np.array([3.39/4, 1.81/3, 0.86/3], dtype=np.float32),
                                   decimal=7)
    print("Test 3 Passed in %s ns" % (time.time_ns() - t))
    arr, t = np.array([-0.0044, -0.0042, -0.0038, -0.0032, -0.0031, -0.0028, -0.0025, -0.0013, -0.0012, -0.001, -0.0008, -0.0003, 0.0001, 0.0006, 0.0009, 0.0012],
                      dtype=np.float32), time.time_ns()
    test.assert_array_almost_equal(compress_width(arr, 0.0007)[0],
                                   np.array([-0.0124/3, -0.0116/4, -0.0043/4, 0.0001, 0.0009], dtype=np.float32),
                                   decimal=3)
    print("Test 4 Passed in %s ns" % (time.time_ns() - t))
    print("All tests successfully passed.")
    return True

def test_compress_ind():
    arr, t = np.array(range(1, 30), dtype=np.float32), time.time_ns()
    test.assert_array_almost_equal(compress_ind(arr, 6)[0],
                                   np.array([3, 8, 13, 18, 23, 27.5], dtype=np.float32),
                                   decimal=7)
    print("Test 1 Passed in %s ns" % (time.time_ns() - t))
    print("All tests successfully passed.")
    return True