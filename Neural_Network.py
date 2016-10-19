import numpy as np
class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 64
        self.outputLayerSize = 1
        self.hiddenLayer1Size = 16
        self.hiddenLayer2Size = 4

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayer1Size)
        self.W2 = np.random.randn(self.hiddenLayer1Size, self.hiddenLayer2Size)
        self.W3 = np.random.randn(self.hiddenLayer2Size, self.outputLayerSize)


if __name__ == '__main__':
    nn = Neural_Network()
    for row in nn.W1:
        print(len(row))