
import math
import time
import numpy as np
from matplotlib import pyplot as plt

def getImageData(filename):
    """Loads image data stored in filename, returns a list of arrays."""

    # with open(filename, 'rb') as fp:
    fp = open(filename, 'rb')
    magicNum = fp.read(4)
    numSamples = int.from_bytes(fp.read(4), byteorder="big")
    numRows = int.from_bytes(fp.read(4), byteorder="big")
    numCols = int.from_bytes(fp.read(4), byteorder="big")

    # Initialize image of size numRows x numCols
    image = np.zeros((numRows, numCols), dtype=np.uint8)

    data = list()

    for inds in range(0, numSamples):
        image = np.zeros((numRows, numCols), dtype=np.uint8)
        for indr in range(0, numRows):
            for indc in range(0, numCols):
                image[indr, indc] = int.from_bytes(fp.read(1), byteorder="big")
        data.append(image)
    fp.close()

    return scaleData(data)

def scaleData(data):
    """Resizes each image in dataset to 8x8 using nearest neighbors scaling.
    Returns a list of vectors of length 64."""

    # Get dimensions of data
    numSamples = len(data)
    (numRows, numCols) = np.shape(data[0])

    # Images are flattened into vectors of lenth 8x8 = 64
    image = np.zeros(( 8 * 8), dtype=np.uint8)

    # Calculate scaling factor
    scale = (8/numRows, 8/numCols)

    out = list()
    for datum in data:
        image = np.zeros(( 8 * 8), dtype=np.uint8)
        for x in range(0, 8):
            for y in range(0, 8):
                image[x*8 + y] = datum[round(x/scale[0]), round(y/scale[1])]

        out.append(image)

    return out


def getLabelData(filename):
    """Loads the label data stored in filename, returns a list of arrays."""

    # with open(filename, 'rb') as fp:
    fp = open(filename, 'rb')
    magicNum = fp.read(4)
    numSamples = int.from_bytes(fp.read(4), byteorder="big")

    data = list()

    for ind in range(0, numSamples):
        data.append(int.from_bytes(fp.read(1), byteorder="big"))
    fp.close()

    y = list()
    for datum in data:
        label = np.zeros(10)
        label[datum] = 1
        y.append(label)

    return y

def combineData(images, labels):
    """Formats data as a list of tuples (x, y), x inputs and y expected
    outputs"""

    data = list(zip(images, labels))

    return data

def getTrainData():
    imageFile = "train-images-idx3-ubyte"
    labelFile = "train-labels-idx1-ubyte"
    images = getImageData(imageFile)
    labels = getLabelData(labelFile)

    data = combineData(images, labels)

    return data

### The following functions are for testing purposes

def test():
    images = timeFunc(getImageData)

    if images[0] is images[90]:
        print("ERROR")

    return images

def timeFunc(func, *args):
    """Prints the execution time of a function and returns the output of the function."""

    # Get time before func
    start = time.clock()
    # Execute func
    out = func(*args)
    # Get time after func
    end = time.clock()
    # Display execution time
    msg = func.__name__ + ": " + "{:.2f}".format(end-start) + "s"
    print(msg)

    return out

def showImage(data, num):
    """Displays the image stored in data[num][0] and its corresponding label."""

    image = data[num][0].reshape((8,8))
    label = str(np.argmax(data[num][1]))

    plt.imshow(image, interpolation="nearest")
    plt.show()
    print("Label: " + str(label))

if __name__ == '__main__':
    print("did it compile?")
    data = getTrainData()