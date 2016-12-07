import numpy as np
import convert_fixed_point

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    step = 0.00024414
    start = 0
    while (start < 5):
        print(convert_fixed_point.Convert(str(sigmoid(start))))
        start = start + step