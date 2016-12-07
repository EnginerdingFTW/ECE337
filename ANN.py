#!/usr/bin/env python

## ----------------------- Part 1 ---------------------------- ##
import numpy as np
import os
import time


## ----------------------- Part 5 ---------------------------- ##
import convert_fixed_point
import FixedToBinary

fileType = "8x8"
class Neural_Network(object):
    def __init__(self):
        #Define Hyperparameters
        self.inputLayerSize = 64
        self.hiddenLayer1Size = 16
        self.hiddenLayer2Size = 8
        self.outputLayerSize = 10

        layerSize, w1, w2, w3 = LoadWeightFromFile()
        if (layerSize == None or layerSize[1] != self.hiddenLayer1Size or layerSize[2] != self.hiddenLayer2Size or layerSize[3] != self.outputLayerSize):
            #Weights (parameters)
            self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayer1Size)
            self.W2 = np.random.randn(self.hiddenLayer1Size,self.hiddenLayer2Size)
            self.W3 = np.random.randn(self.hiddenLayer2Size, self.outputLayerSize)
        else:
            self.W1 = w1
            self.W2 = w2
            self.W3 = w3

    def forward(self, X):
        self.x = X
        #Propogate inputs though network
        # print("X shape = " + str(X.shape))
        # print("W1 shape = " + str(self.W1.shape))
        self.z2 = np.dot(X, self.W1)
        # print("z2 shape = " + str(self.z2.shape))
        self.a2 = self.sigmoid(self.z2)
        # print("a2 shape = " + str(self.a2.shape))
        self.z3 = np.dot(self.a2, self.W2)
        # print("z3 shape = " + str(self.z3.shape))
        self.a3 = self.sigmoid(self.z3)
        # print("a3 shape = " + str(self.a3.shape))
        self.z4 = np.dot(self.a3, self.W3)
        # print("z4 shape = " + str(self.z4.shape))
        self.yHat = self.sigmoid(self.z4)
        # print("yHat shape = " + str(yHat.shape))
        return self.yHat

    def forwardReturnAll(self, X):
        #Propogate inputs though network
        # print("X shape = " + str(X.shape))
        # print("W1 shape = " + str(self.W1.shape))
        self.z2 = np.dot(X, self.W1)
        # print("z2 shape = " + str(self.z2.shape))
        self.a2 = self.sigmoid(self.z2)
        # print("a2 shape = " + str(self.a2.shape))
        self.z3 = np.dot(self.a2, self.W2)
        # print("z3 shape = " + str(self.z3.shape))
        self.a3 = self.sigmoid(self.z3)
        # print("a3 shape = " + str(self.a3.shape))
        self.z4 = np.dot(self.a3, self.W3)
        # print("z4 shape = " + str(self.z4.shape))
        yHat = self.sigmoid(self.z4)
        # print("yHat shape = " + str(yHat.shape))
        return self.a2, self.a3, yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

        # x_size, y_size = z.shape
        # for x in range(0, x_size):
        #     for y in range(0, y_size):
        #         if (z[x][y] < 0):
        #             z[x][y] = 0
        #         else:
        #             z[x][y] = z[x][y]
        # return z

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

        # x_size, y_size = z.shape
        # for x in range(0, x_size):
        #     for y in range(0, y_size):
        #         if (z[x][y] >= 0):
        #             z[x][y] = 1
        #         else:
        #             z[x][y] = 0
        # return z

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        #print("y.shape = " + str(y.shape))
        #print("yhat.shape = " + str(self.yHat.shape))
        temp = y - self.yHat
        J = 0.5*sum(temp**2)
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        # print("yhat.shape = " + str(self.yHat.shape))
        # print("z4.shape = " + str(self.z4.shape))
        # print("a3.T.shape = " + str(self.a3.T.shape))

        delta4 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z4))
        dJdW3 = np.dot(self.a3.T, delta4)

        delta3 = np.dot(delta4, self.W3.T)*self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2, dJdW3

    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayer1Size * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayer1Size))
        W2_end = W1_end + self.hiddenLayer1Size*self.hiddenLayer2Size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayer1Size, self.hiddenLayer2Size))
        W3_end = W2_end + self.hiddenLayer2Size*self.outputLayerSize
        self.W3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayer2Size, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel()))

    def SaveWeights(self):
        SaveWeightsToFile([self.W1, self.W2, self.W3], [self.inputLayerSize, self.hiddenLayer1Size, self.hiddenLayer2Size, self.outputLayerSize])

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0

        #Return Params to original value:
        N.setParams(paramsInitial)
        return numgrad

## ----------------------- Part 6 ---------------------------- ##
# from scipy import optimize

class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        self.learningRate = 0.1

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad

    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}

        self.myOptimize(params0, X, y)

        #cost, grad = self.costFunctionWrapper(params0, X, y)
        #_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
        #                         args=(X, y), options=options, callback=self.callbackF)

        #self.N.setParams(_res.x)
        #self.optimizationResults = _res


    def myOptimize(self, params0, X, y):
        cost, grad = self.costFunctionWrapper(params0, X, y)
        while (np.any(np.greater(cost, 0.2))):
            params0 = params0 - self.learningRate * grad
            self.N.setParams(params0)
            cost, grad = self.costFunctionWrapper(params0, X, y)

def SaveWeightsToFile(list_weights, layerSizes):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    fp = open(dir_path + "/shapeWeights" + fileType + ".txt", 'w')
    for item in layerSizes:
        fp.write(str(item) + "\n")
    for weights in list_weights:
        for row in weights:
            for item in row:
                temp = "%.9f" % item
                fp.write(temp + "\n")
    fp.close()

def LoadWeightFromFile():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path + "/shapeWeights" + fileType + ".txt"
    if (not os.path.isfile(dir_path)):
        return None, None, None, None
    fp = open(dir_path, 'r')
    lines = fp.readlines()
    fp.close()

    k = 0
    layerSize = []
    weights = []
    for line in lines:
        if (k < 4):
            line = line.strip('\n')
            layerSize.append(int(line))
        else:
            weights.append(float(line))
        k = k + 1


    W1 = []
    k = 0
    for i in range(0, layerSize[0]):
        temp = []
        for j in range(0, layerSize[1]):
            temp.append(weights[k])
            k = k + 1
        W1.append(temp)
    W1 = np.asarray(W1)
    # print(W1.shape)

    W2 = []
    for i in range(0, layerSize[1]):
        temp = []
        for j in range(0, layerSize[2]):
            temp.append(weights[k])
            k = k + 1
        W2.append(temp)
    W2 = np.asarray(W2)
    # print(W2.shape)

    W3 = []
    for i in range(0, layerSize[2]):
        temp = []
        for j in range(0, layerSize[3]):
            temp.append(weights[k])
            k = k + 1
        W3.append(temp)
    W3 = np.asarray(W3)
    # print(W3.shape)

    return layerSize, W1, W2, W3



def WriteArrayToFile(array, filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + "/testcases" + fileType + "/" + filename
    fp = open(path, 'w')
    for element in array:
        temp = "%.9f" % element
        fp.write(temp + "\n")
    fp.close()
    convert_fixed_point.ConvertToFixedPoint(path)
    os.remove(path) #removes the files that were generated for standard floating point, leaving just the fixed point

if __name__ == '__main__':
    choice2 = input("Is this 8x8 or 4x4 (1/0)?: ")
    choice = input("Are you training (1/0)?: ")
    size = 8
    if (choice2 ==  0):
        fileType = "4x4"
        size = 4

    start = time.time()
    lineCount = 0
    while lineCount < 540000:


        dir_path = os.path.dirname(os.path.realpath(__file__))
        fp = open(dir_path + "/MNISTdataset" + fileType + ".txt", 'r')
        lines2 = fp.readlines()
        fp.close()

        temp_final = []
        y_final = []
        k = 1


        k = 1
        current_tri = []
        for line in lines2:
            lineCount = lineCount + 1
            if (k % (size + 1) != 0):
                for item in line.split('\t')[0:size]:
                    current_tri.append(float(item))
            else:
                temp_final.append(np.asarray(current_tri))
                list_temp = []
                for item in line.split('\t')[0:10]:
                    list_temp.append(float(item))
                y_final.append(list_temp)
                current_tri = []

            if (k == (size + 1) * 10):
                break
            k = k + 1



        temp_final = np.asarray(temp_final)
        y_final = np.asarray(y_final)
        arr = np.asarray(temp_final)
        nn = Neural_Network()
        nn.inputLayerSize = size * size

        if (choice == 0):
            temp_final[0] = np.zeros(size * size)
            break

        print("before training:" + "  line count = " + str(lineCount))
        out = nn.forward(arr)
        t = trainer(nn)
        t.train(arr, np.asarray(y_final))
    print("after training:")
    out = nn.forward(arr)
    #print(nn.W3)
    #print(out)
    for i in range(0, 10):
        print("i = " + str(i) + ":")
        print(out[i])

    end = time.time()
    print("time = " + str(end - start))
    nn.SaveWeights()

    #testcase file generation
    k = 0
    for array in nn.x:
        name_temp = "t" + str(k) + "in.txt"
        name_temp2 = "t" + str(k) + "in_fixed.txt"
        WriteArrayToFile(array,name_temp)
        FixedToBinary.ConvertToBinary(os.path.dirname(os.path.realpath(__file__)) + "/testcases" + fileType + "/" + name_temp2)
        k = k + 1
    k = 0
    for array in nn.a2:
        WriteArrayToFile(array,"t" + str(k) + "lay1.txt" )
        k = k + 1
    k = 0
    for array in nn.a3:
        WriteArrayToFile(array,"t" + str(k) + "lay2.txt" )
        k = k + 1
    k = 0
    for array in nn.yHat:
        WriteArrayToFile(array,"t" + str(k) + "layOut.txt" )
        k = k + 1
    for item in nn.W3:
        print(item)

