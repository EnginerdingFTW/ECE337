import numpy as np
import os

def GenerateSquares():
    list_squares = []
    for x in range(0, 8):
        for y in range(0, 8):
            list = GenerateSquaresFromPoint(x, y)
            if (list != []):
                list_squares.extend(list)
    return list_squares

def GenerateSquaresFromPoint(x, y):
    list = []
    for size in range(1, 9):
        square = np.zeros((8, 8))
        if (x + size < 8 and y + size < 8):
            for i in range(x, x + size + 1):
                square[i][y] = 1
                square[i][y + size] = 1
            for j in range(y, y + size + 1):
                square[x][j] = 1
                square[x + size][j] = 1
            list.append(square)
    return list


def WriteSquareToFile(arr, filepath_name):
    fp = open(filepath_name, 'w')
    for square in arr:
        s = ""
        for row in square:
            for item in row:
                s = s + str(item) + "\t"
            s = s + "\n"
        s = s + "\n"
        fp.write(s)

    fp.close()



if __name__ == '__main__':
    list = GenerateSquares()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    WriteSquareToFile(list, dir_path + "/testSquares.txt")
