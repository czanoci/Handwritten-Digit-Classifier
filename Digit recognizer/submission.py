import collections
import math
from collections import Counter
from util import *
from mnist import *
from operator import itemgetter


def featureDict(l):
    counter = Counter()
    for i in xrange(len(l)):
        if l[i] != 0:
            counter[i] = l[i]
    return counter

def learnPredictor(trainExamples, trainLabels, testExamples, testLabels):
    numClasses = 10
    numIters = 11
    eta = 0.1

    weights = []
    for i in xrange(numClasses):
        weights.append(Counter())

    def predictor(x):
        l = []
        x = featureDict(x)
        for i in xrange(numClasses):
            l.append(dotProduct(weights[i], x))
        index, element = max(enumerate(l), key=itemgetter(1))
        return index

    def sgd(trainExamples, trainLabels, weights, numClasses, eta, numIters, testExamples, testLabels):
        for t in xrange(numIters):
            etaNew = 1.0*eta/math.sqrt(t+1)
            for i in xrange(len(trainExamples)):
                x = featureDict(trainExamples[i])
                correctLabel = trainLabels[i]
                products = []
                for j in xrange(numClasses):
                    products.append( dotProduct(weights[j], x) )
                predictedLabel, element = max(enumerate(products), key=itemgetter(1))
                if predictedLabel != correctLabel:
                    increment(weights[correctLabel], etaNew, x)
                    increment(weights[predictedLabel], -etaNew, x)
            print "Test Error:" + str(evaluatePredictor(testExamples, testLabels, predictor))

    sgd(trainExamples, trainLabels, weights, numClasses, eta, numIters, testExamples, testLabels)

mndata = MNIST('.')
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_testing()

learnPredictor(train_img, train_label, test_img, test_label)
