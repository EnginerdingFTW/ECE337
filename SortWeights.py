import os
import numpy as np



def LoadWeightFromFile():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path + "/shapeWeights.txt"
    if (not os.path.isfile(dir_path)):
        return None, None, None, None
    fp = open(dir_path, 'r')
    lines = fp.readlines()
    fp.close()

    temp = []
    last = 10000
    for line in lines:
        temp.append(float(line))
    temp.sort()

    min_size = 10000
    for line in temp:
        t = line
        if (abs(last - t) < min_size):
            min_size = last - t
        last = t

    return temp, min_size

if __name__ == '__main__':
    arr, min = LoadWeightFromFile()
    for item in arr:
        print(item)
    print("min_size = " + str(min))

