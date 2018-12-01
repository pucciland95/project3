from Networks import Network
from database_loader import load_data
from Networks import sigmoid, reLu, tanh, leakyReLu
from time import time


[trainingData, validationData, testData] = load_data()
trainingData = list(trainingData)
validationData = list(validationData)
testData = list(testData)

"""
cont = 0
lamndaList = [0, 1, 2, 0, 1, 2, 0, 1, 2]
learningParameter = [2, 2, 2, 3, 3, 3, 4, 4, 4]
contLearning = 0

##################################################EXPERIMENTS DONE WITH THE sigmoid with no hidden layers###########################################
while contLearning < len(learningParameter):

    net = Network([784, 10], [None, sigmoid])
    startingTime = time()
    net.training(trainingData, 20, 50, learningParameter[contLearning], lamndaList[contLearning]/len(trainingData), validationData)
    finalTime = time() - startingTime
    contLearning += 1

    print("The testing accuracy is equal to: " + str(net.evaluate(testData)/len(testData)*100))
    print("The requested training time has been: ", finalTime)
    print("\n\n\n")

##################################################EXPERIMENTS DONE WITH THE sigmoid with 1 hidden layer###########################################
lamndaList = [0, 1, 2, 0, 1, 2, 0, 1, 2]
learningParameter = [2, 2, 2, 3, 3, 3, 4, 4, 4]
contLearning = 0
while contLearning < len(learningParameter):

    net = Network([784, 30, 10], [None, sigmoid, sigmoid])
    startingTime = time()
    net.training(trainingData, 20, 50, learningParameter[contLearning], lamndaList[contLearning]/len(trainingData), validationData)
    finalTime = time() - startingTime
    contLearning += 1

    print("The testing accuracy is equal to: " + str(net.evaluate(testData)/len(testData)*100))
    print("The requested training time has been: ", finalTime)
    print("\n\n\n")

##################################################EXPERIMENTS DONE WITH THE tanh###########################################
lamndaList = [1, 2, 1, 2, 1, 2]
learningParameter = [1.2, 1.2, 0.9, 0.9, 1.5, 1.5]
contLearning = 0
while contLearning < len(learningParameter):

    net = Network([784, 30, 10], [None, tanh, tanh])
    startingTime = time()
    net.training(trainingData, 50, 50, learningParameter[contLearning], lamndaList[contLearning]/len(trainingData), validationData)
    finalTime = time() - startingTime
    contLearning += 1

    print("The testing accuracy is equal to: " + str(net.evaluate(testData)/len(testData)*100))
    print("The requested training time has been: ", finalTime)
    print("\n\n\n")
"""
##################################################EXPERIMENTS DONE WITH THE leakyReLu###########################################
cont = 0
lamndaList = [5, 5]
learningParameter = [0.001, 0.01]
contLearning = 0
while contLearning < len(learningParameter):

    net = Network([784, 30, 10], [None, leakyReLu, leakyReLu])
    startingTime = time()
    net.training(trainingData, 400, 50, learningParameter[contLearning], lamndaList[contLearning]/len(trainingData), validationData)
    finalTime = time() - startingTime
    contLearning += 1

    print("The testing accuracy is equal to: " + str(net.evaluate(testData)/len(testData)*100))
    print("The requested training time has been: ", finalTime)
    print("\n\n\n")
