class NeuralNode(object):
    neuralConnections = []

    def __init__(self):
        self.neuralConnections = []

    def addNeuralConnection(self, newConnection):
        self.neuralConnections.append(newConnection)

    def getListOfConnections(self):
        return self.neuralConnections

    def CalculateOutput(self):
        temp = 0;
        for connection in self.neuralConnections:
            temp = temp + connection.GetValueOut()
        return temp




class NeuralConnection(object):
    weight = 0
    inputNode = None

    def __init__(self, start, w):
        self.inputNode = start
        self.weight = w

    def GetValueOut(self):
        temp = None
        if (self.inputNode != None):
            if (type(self.inputNode) == type(NeuralNode())):
                temp = self.weight * self.inputNode.CalculateOutput()
            else:
                temp = self.weight * self.inputNode
        return temp

    def SetWeight(self, newWeight):
        self.weight = newWeight

    def GetWeight(self):
        return self.weight


class NeuralNetworkLayer(object):
    size = 8
    inputMatrix = []
    layerNodes = [[]]

    def __init__(self, array):
        self.layers = 3
        self.inputMatrix = array
        for temp in array:
            self.size = len(temp)
            break
        if (self.size == 2):
             self.FinalLayer()
        else:
             self.CreateLayer()

    def CreateLayer(self):
        arr = []
        for x in range(0, self.size / 2):
            temp = []
            for y in range(0, self.size / 2):
                neuralNode = NeuralNode()
                for i in range(2*x - 1, (2*x + 2) + 1):
                    for j in range(2*y - 1, (2*y + 2) + 1):
                        if (self.VerifyInArray(i, j)):
                            stringtemp = str(i) + " : " + str(j)
                            neuralNode.addNeuralConnection(NeuralConnection(self.inputMatrix[i][j], 0.5))
                temp.append(neuralNode)
            arr.append(temp)
        self.layerNodes = arr

    def VerifyInArray(self, a, b):
        temp = True
        if (a < 0 or a >= self.size):
            temp = False
        if (b < 0 or b >= self.size):
            temp = False
        return temp

    def FinalLayer(self):
        neuralNode = NeuralNode()
        for x in range(0, 2):
            for y in range(0, 2):
                neuralNode.addNeuralConnection(NeuralConnection(self.inputMatrix[x][y], 0.5))
        self.layerNodes = [neuralNode]


class NeuralNetwork(object):
    size = 8
    layers = 3
    inputMatrix = []
    layerMatrix = []
    output = 0

    def __init__(self, input):
        self.layers = 3
        self.size = 8
        self.inputMatrix = input
        self.layerMatrix.append(NeuralNetworkLayer(input))
        self.layerMatrix.append(NeuralNetworkLayer(self.layerMatrix[0].layerNodes))
        self.layerMatrix.append(NeuralNetworkLayer(self.layerMatrix[1].layerNodes))


if __name__ == '__main__':
    arr = []
    for x in range(0, 8):
        temp = []
        for y in range(0, 8):
            temp.append(1)
        arr.append(temp)
    for x in arr:
        print(x)

    nn = NeuralNetwork(arr)
    lay = []
    for layer in nn.layerMatrix:
        m = []
        if (len(layer.layerNodes) > 1):
            for x in layer.layerNodes:
                temp = []
                for y in x:
                    temp.append(len(y.neuralConnections))
                m.append(temp)
            lay.append(m)
        else:
            lay.append(len(layer.layerNodes[0].neuralConnections))
    for m in lay:
        print("Layer:")
        if (type(m) == type(NeuralNetworkLayer)):
            for x in m:
                print(x)
        else:
            print(m)
    print(layer.layerNodes[0].CalculateOutput())









    # layer = NeuralNetworkLayer(arr)
    # value1 = []
    # print("-------------")
    # for x in layer.layerNodes:
    #     temp1 = []
    #     for y in x:
    #         temp1.append(len(y.neuralConnections))
    #     value1.append(temp1)
    # coor = []
    # for x in value1:
    #    print(x)
    #
    #
    # print("***************************************")
    # arr = []
    # for x in range(0, 2):
    #     temp = []
    #     for y in range(0, 2):
    #         temp.append(1)
    #     arr.append(temp)
    # for a in arr:
    #     print(a)
    #
    # layer = NeuralNetworkLayer(arr)
    # print("-------------")
    # print(len(layer.layerNodes[0].neuralConnections))

