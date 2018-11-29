import numpy as np
import random
from copy import deepcopy


class Network(object):
    def __init__(self, sizes, activationFncs):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.activationFcns = activationFncs


    def inference(self, x):
        for i in range(self.num_layers - 1):  # i represents the layer we currently are
            zList = []
            weight = self.weights[i]
            bias = self.biases[i]
            actFuncActualLayer = self.activationFcns[i + 1]  # actiavation function at the layer i (the +1 is added because we don' t want the initial layer with None as actFunc to be part of it)

            for y in range(len(weight)):
                zList.append(np.dot(x, weight[y]) + float(bias[y]))

            for k in range(len(zList)):
                zList[k] = actFuncActualLayer(zList[k])

            x = []
            for k in zList:
                x.append(k)

        return np.array(x)

    def training(self, trainData, T, n, alpha, lmbda, validationData, earlyStop = 5):
        cont = 0
        contAccuracy = 0

        backUpWeight = deepcopy(self.weights)
        backUpBias = deepcopy(self.biases)
        bestAccuracy = self.evaluate(validationData) / len(validationData) * 100

        while cont < T:
            ######Division of training data in batch ###############
            batchList = []
            random.shuffle(trainData)

            batchElem = []
            for x in trainData:
                batchElem.append(x)
                if len(batchElem) == n:
                    batchList.append(batchElem)
                    batchElem = []

            for batch in batchList:
                self.updateWeights(batch, alpha, lmbda)

            accuracyActualEpoch = self.evaluate(validationData) / len(validationData) * 100

            if accuracyActualEpoch < bestAccuracy:
                contAccuracy += 1
                if contAccuracy > earlyStop:
                    print("Early Stopping occured")
                    self.weights = backUpWeight #Both the weights and the biases go back to the best values where a better accuracy was found
                    self.biases = backUpBias   #Both the weights and the biases go back to the best values where a better accuracy was found
                    break
            else:
                backUpBias = deepcopy(self.biases)
                backUpWeight = deepcopy(self.weights)
                bestAccuracy = accuracyActualEpoch
                contAccuracy = 0 #if the accuracy increases the conter for the early sttopping get changed

            cont += 1

        print("The evaluetion accuracy is: ", bestAccuracy)
        self.weights = backUpWeight
        self.biases = backUpBias

    def updateWeights(self, batch, alpha, lmbda):
        gradientList = []

        for x in batch:
            gradientList.append(self.backprop(x[0], x[1]))

        averagedWeight = [np.zeros_like(b) for b in self.weights]
        averagedBias = [np.zeros_like(b) for b in self.biases]

        for x in gradientList: #Sum all over the gradients obtained by backpropagation
            averagedWeight = [sum(y) for y in zip(x[0], averagedWeight)]
            averagedBias = [sum(y) for y in zip(x[1], averagedBias)]

        for x in range(len(averagedWeight)):
            averagedWeight[x] = averagedWeight[x] / len(batch)

        for x in range(len(averagedBias)):
            averagedBias[x] = averagedBias[x] / len(batch)


        cont = 0
        while cont <= len(averagedWeight) - 1:
            self.weights[cont] = self.weights[cont] - alpha * averagedWeight[cont] - alpha * lmbda * self.weights[cont]
            cont += 1

        cont = 0
        while cont <= len(averagedBias) - 1:
            self.biases[cont] = self.biases[cont] - alpha * averagedBias[cont]
            cont += 1

    def backprop(self, x, y):
        x = np.reshape(x, x.shape[:-1])
        y = np.reshape(y, y.shape[:-1])

        aList = []  # List containing all the outputs of the neurons of the neural network
        zList = [] #######TEST################

        ################################## Inference ###############################
        a = x
        for k in range(0, len(self.sizes) -1):
            bias = np.reshape(self.biases[k], self.biases[k].shape[:-1])
            z = np.dot(self.weights[k], a) + bias
            zList.append(z) #######TEST################
            a = self.activationFcns[k + 1](z)
            aList.append(a)
        ###########################################################################
        zList.insert(0, x)
        aList.insert(0, x)


        gradient = [0] * (self.num_layers - 1)
        bias = [0] * (self.num_layers - 1)

        dErr = dSquaredLoss(aList[len(aList) - 1], y)  # List containing all the derivatives of the error
        actFunct = self.activationFcns[len(self.sizes) - 1]  # I take the last activation function because at the first iteration we are in the first layer
        #dActFunct = derActFncs(actFunct, aList[len(aList) - 1])  # Derivative of the activation function evalueted at a given a #######DA RIMUOVERE PER IL TEST################
        dActFunct = derActFncs(actFunct, zList[len(zList) - 1])  # Derivative of the activation function evalueted at a given z #######TEST################

        np.array(dActFunct)
        deltaL = dActFunct * dErr
        flag = 0  # flag that tell us if we are at the first iteration

        for j in range(self.num_layers - 2, -1, -1):
            if flag != 0:
                deltaL = deltaPrevLayer

            # costruction of the previous deltaL-1
            #deltaPrevLayer = np.sum(self.weights[j].transpose() * deltaL, axis=1) * derActFncs(self.activationFcns[j + 1], aList[j])
            deltaPrevLayer = np.sum(self.weights[j].transpose() * deltaL, axis=1) * derActFncs(self.activationFcns[j + 1], zList[j]) #######TEST################
            flag = 1
            mom = []

            for i in deltaL:
                mom.append([i])
            bias[j] = np.array(mom)

            gradient[j] = np.outer(deltaL, aList[j])

        return (gradient, bias)

    def evaluate(self, data):
        corrPred = 0
        for x in data:
            transposeData = np.reshape(x[0], x[0].shape[:-1])
            result = self.inference(transposeData)
            indexMax = np.argmax(result)
            if indexMax == x[1]:
                corrPred += 1
        return corrPred


# activation functions together with their derivative functions:
def dSquaredLoss(a, y):
    return -np.subtract(y, a)


#############Activation functions definitions#########################
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return (1 - np.exp(-2 * z)) / (1 + np.exp(-2 * z))


def reLu(z):
    return max(0, z)


################Derivates definitions#######################
"""
def sigmoid_prime(a):
    return a * (1 - a) #######DA RIMUOVERE PER IL TEST################
"""
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z)) #######TEST################

"""
def tanhPrime(a):
    return 1 - np.power(a, 2) #######DA RIMUOVERE PER IL TEST################
"""
def tanhPrime(z):
    return 1 - np.power(tanh(z), 2) #######TEST################

"""
def reLuPrime(a):
    result = []

    for x in a:
        if x > 0:
            result.append(1)
        else:
            result.append(0)

    return np.array(result)  #######DA RIMUOVERE PER IL TEST################
"""
def reLuPrime(z):
    z[z > 0] = 1
    z[z < 0] = 0
    return z #######TEST################


def derActFncs(actFunc, a):
    if actFunc == sigmoid:
        return sigmoid_prime(a)

    if actFunc == tanh:
        return tanhPrime(a)

    if actFunc == reLu:
        return reLuPrime(a)

    else:
        print("No act function found!")