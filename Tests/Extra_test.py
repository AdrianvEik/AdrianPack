import numpy as np
import numpy.testing
from Extra import compress_array

# TODO: Tests for both ind and width compressing
# TODO: Write arrays out manually and test
def test_Compress_array(test_ind, tests):
    fail = 0
    deviation = []
    test_list = [np.random.random(np.random.randint(1000, 10000, 1)[0]) for _ in range(tests)]
    compress = compress_array(test_list, width_ind=test_ind)
    for i in compress:
        if len(i) == test_ind:
            pass
        else:
            deviation.append(np.absolute(len(i) - test_ind))
            fail += 1

    return (1 - fail/tests) * 100, np.average(deviation)

if __name__ == "__main__":
    print(test_Compress_array(100, 3))