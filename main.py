from Networks import Network
from database_loader import load_data
from Networks import sigmoid, reLu, tanh
from time import time


[trainingData, validationData, testData] = load_data()
trainingData = list(trainingData)
validationData = list(validationData)
testData = list(testData)


accuracy = 0
cont = 0
lamndaList = [0, 1, 0, 1, 0, 1]
learningParameter = [2, 2, 3, 3, 4, 4]
contLearning = 0
while contLearning < len(learningParameter):

    net = Network([784, 30, 10], [None, tanh, tanh])  # La reLu e' baggata!!
    startingTime = time()
    net.training(trainingData, 20, 50, learningParameter[contLearning], lamndaList[contLearning]/len(trainingData), validationData)
    finalTime = time() - startingTime
    contLearning += 1

    print("The testing accuracy is equal to: " + str(net.evaluate(testData)/len(testData)*100))
    print("The requested training time has been: ", finalTime)

#b = net.backprop([0.05, 0.1], [0.01, 0.99])
#print(b)

#net = Network([2, 3, 2], [None, sigmoid, sigmoid])
#net.updateWeights([(np.array([[0.05], [0.1]]), np.array([[0.01], [0.99]]))], 0.5, 0)

# print(net.evaluate([([0.05, 0.1], [0, 1])]))
#trainingDataset = [([0.05, 0.1],[0.01, 0.99])]
#validationData = [([0.05, 0.1],[0, 1])]
#print(net.training(trainingDataset, 5, 1, 0.5, 0, validationData))