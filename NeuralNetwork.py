import math
import random
import os
import numpy as np

class NeuralNode(object):
    neuralConnections = []
    pointer = None

    def __init__(self, p):
        self.pointer = p
        self.neuralConnections = []
        self.counter = 0

    def addNeuralConnection(self, newConnection):
        self.neuralConnections.append(newConnection)

    def getListOfConnections(self):
        return self.neuralConnections

    def CalculateOutput(self):
        return self.ActivationFunction(self.GetZ())

    def GetZ(self):
        temp = 0;
        for connection in self.neuralConnections:
            temp = temp + connection.GetValueOut()
        return np.asfarray(temp)

    def GetW(self):
        temp = []
        for con in self.neuralConnections:
            temp.append(con.weightIndex)
        return np.asfarray(temp)

    def GetA(self):
        return np.asfarray(self.CalculateOutput())

    def ActivationFunction(self, input):
        return 1 / (1 + np.exp(-input))

    def ActivationFunctionDer(self, input):
        return np.exp(-input) / ((1 + np.exp(-input))**2)




class NeuralConnection(object):
    weightIndex = 0
    inputNode = None
    pointer = None

    def __init__(self, start, wIndex, p):
        self.pointer = p
        self.inputNode = start
        self.weightIndex = wIndex

    def GetValueOut(self):
        temp = None
        if (self.inputNode != None):
            if (type(self.inputNode) == type(NeuralNode(self.pointer))):
                temp = self.pointer.GetWeight(self.weightIndex) * self.inputNode.CalculateOutput()
            else:
                temp = self.pointer.GetWeight(self.weightIndex) * self.inputNode
        return temp

    def SetWeight(self, newWeight):
        self.pointer.ChangeWeight(self.pointer.GetWeight(self.weightIndex), newWeight)

    def GetWeight(self):
        return self.pointer.GetWeight(self.weightIndex)


class NeuralNetworkLayer(object):
    size = 8
    inputMatrix = []
    layerNodes = [[]]
    pointer = None

    def __init__(self, array, p):
        self.pointer = p
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
                neuralNode = NeuralNode(self.pointer)
                for i in range(2*x - 1, (2*x + 2) + 1):
                    for j in range(2*y - 1, (2*y + 2) + 1):
                        if (self.VerifyInArray(i, j)):
                            neuralNode.addNeuralConnection(NeuralConnection(self.inputMatrix[i][j], self.pointer.LoadWeightInd(), self.pointer))
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
        neuralNode = NeuralNode(self.pointer)
        for x in range(0, 2):
            for y in range(0, 2):
                neuralNode.addNeuralConnection(NeuralConnection(self.inputMatrix[x][y], self.pointer.LoadWeightInd(), self.pointer))
        self.layerNodes = [neuralNode]

    def getZOfLastLay(self):
        temp = []
        if (len(self.layerNodes) == 1):
            for con in self.layerNodes[0].neuralConnections:
                temp.append(con.inputNode.GetZ())
            return np.asfarray(temp)

        for row in self.layerNodes:
            for node in row:
                for con in node.neuralConnections:
                    temp.append(con.inputNode.GetZ())
        return np.asfarray(temp)

    def getAOfLastLay(self):
        temp = []
        if (len(self.layerNodes) == 1):
            for con in self.layerNodes[0].neuralConnections:
                temp.append(con.inputNode.CalculateOutput())
            return np.asfarray(temp)

        for row in self.layerNodes:
            for node in row:
                for con in node.neuralConnections:
                    temp.append(con.inputNode.CalculateOutput())
        return np.asfarray(temp)

    def getW(self):
        temp = []
        for row in self.layerNodes:
            for node in row:
                for connection in node.neuralConnections:
                    temp.append(connection.weightIndex)
        return np.asfarray(temp)

class NeuralNetwork(object):
    size = 8
    layers = 3
    pointer = None
    inputMatrix = []
    layerMatrix = []
    output = 0
    yHat = None #defined later as array of size # inputs


    def __init__(self, input, p):
        self.layers = 3
        self.size = 8
        self.pointer = p
        self.inputMatrix = input

    def ActivationFunction(self, input):
        return 1 / (1 + np.exp(-input))

    def ActivationFunctionDer(self, input):
        return np.exp(-input) / ((1 + np.exp(-input))**2)

    def forward(self):
        return self.layerMatrix[2].layerNodes[0].CalculateOutput()

    def forward(self, X):
        layer1 = NeuralNetworkLayer(X, self.pointer)
        layer2 = NeuralNetworkLayer(layer1.layerNodes, self.pointer)
        outputNode = NeuralNetworkLayer(layer2.layerNodes, self.pointer)

        self.layerMatrix.append(layer1)
        self.layerMatrix.append(layer2)
        self.layerMatrix.append(outputNode)

        self.W1 = layer1.getW()
        self.Z2 = layer2.getZOfLastLay()
        self.A2 = layer2.getAOfLastLay()
        print("length of W1: " + str(len(self.W1)))
        # print("length of Z2: " + str(len(self.Z2)))
        # print("length of A2: " + str(len(self.A2)))

        self.W2 = layer2.getW()
        self.Z3 = outputNode.getZOfLastLay()
        self.A3 = outputNode.getAOfLastLay()
        print("length of W2: " + str(len(self.W2)))
        # print("length of Z3: " + str(len(self.Z3)))
        # print("length of A3: " + str(len(self.A3)))

        self.W3 = outputNode.layerNodes[0].GetW()
        self.Z4 = outputNode.layerNodes[0].GetZ()
        self.yHat = outputNode.layerNodes[0].GetA()
        print("length of W3: " + str(len(self.W3)))
        # print("length of Z4: " + str(self.Z4))
        # print("length of yHat: " + str(self.yHat))

        return self.yHat


    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        if (type(y) == type(0.5)  or type(y) == type(1)):
            J = 0.5 * ((y - self.yHat)**2)
        else:
            J = 0.5 * sum((y - self.yHat)**2)
        return J

    def costFunctionPrime(self, X, Y):
        self.yHat = self.forward(X)

        delta4 = np.multiply(-(y-self.yHat), self.ActivationFunctionDer(self.z4))
        dJdW3 = np.dot(self.a3.T, delta4)

        w3 = self.pointer.GetWeight(self.W3)
        delta3 = delta4 * np.multiply(w3.T, self.ActivationFunctionDer(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        w2 = self.pointer.GetWeight(self.W2)
        delta2 = delta3 * np.multiply(w2.T, self.ActivationFunctionDer(self.z2))
        dJdW1 = np.dot(self.X.T, delta2)

        return dJdW1, dJdW2, dJdW3


class Pointer(object):
    weights = np.asfarray(np.zeros(236))
    i = 0
    loadWeightInd = 0
    weightsLen = 236

    def __init__(self):
        self.weightsLen = 236
        return

    def AddWeight(self, node):
        self.weights[self.i] = node
        self.i = self.i + 1
        return self.i - 1

    def GetWeight(self, index):
        if (type(index) == type(np.array([1, 1]))):
            temp = []
            for element in index:
                temp.append(self.weights[element])
            return np.asfarray(temp)
        else:
            return self.weights[index]

    def ChangeWeight(self, index, value):
        self.weights[index] = value

    def LoadWeightInd(self):
        self.loadWeightInd = self.loadWeightInd + 1
        return self.loadWeightInd - 1

    def GenerateRandomWeights(self):
        for x in range(0, self.weightsLen):
            self.AddWeight(random.randrange(-5, 5))

    def SaveWeightsToFile(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # print(dir_path)
        fp = open(dir_path + "/shapeWeights.txt", 'w')
        for item in self.weights:
            temp = "%.9f" % item
            fp.write(temp + "\n")
        fp.close()

    def LoadWeightFromFile(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        fp = open(dir_path + "/shapeWeights.txt", 'r')
        lines = fp.readlines()
        fp.close()
        if (len(lines) < 2):
            self.GenerateRandomWeights()
            return
        k = 0
        for line in lines:
            self.weights[k] = float(line)
            k = k + 1

    def UpdateWeights(self, newWeights, sel):
        if (sel == 1):
            #W1 is indicies 0-195
            for k in range(0, 196):
                self.ChangeWeight(k, newWeights[k])
        elif (sel == 2):
            #W2 is indices 196 - 231
            for k in range(196, 232):
                    self.ChangeWeight(k, newWeights[k - 196])
        elif (sel == 3):
            #W3 is indices 232 - 235
            for k in range(232, 236):
                    self.ChangeWeight(k, newWeights[k - 232])



# if __name__ == '__main__':
#     arr = [[1, 1, 1, 1, 1, 1, 1, 1],
#            [1, 0, 0, 0, 0, 0, 0, 1],
#            [1, 0, 0, 0, 0, 0, 0, 1],
#            [1, 0, 0, 0, 0, 0, 0, 1],
#            [1, 0, 0, 0, 0, 0, 0, 1],
#            [1, 0, 0, 0, 0, 0, 0, 1],
#            [1, 0, 0, 0, 0, 0, 0, 1],
#            [1, 1, 1, 1, 1, 1, 1, 1]]
#     for x in arr:
#         print(x)
#     p = Pointer()
#     p.LoadWeightFromFile()
#     nn = NeuralNetwork(arr, p)
#     nn.forward(arr)
#     # a, b, c = nn.costFunctionPrime(arr, 1)
#     p.SaveWeightsToFile()








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

