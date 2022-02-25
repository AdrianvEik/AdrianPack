import numpy as np
import numpy.testing
from Extra import Compress_array

# TODO: Tests for both ind and width compressing
# TODO: Write arrays out manually and test
def test_Compress_array():
    r_arr = np.random.random(len(range(0, 30)))
    print(r_arr)
    print(Compress_array([np.array(range(0, 30)), r_arr], 5))

if __name__ == "__main__":
    test_Compress_array()