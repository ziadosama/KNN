
# coding: utf-8

#http://cs231n.github.io/classification/
import random
import math
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import NearestNeighbor as NN

#constants
K_FOLDS=5



#functions
def rgb2gray(xtr):
    return np.dot(xtr[...,:3], [0.299, 0.587, 0.114])

def conc(foldsX,index):    #concatenate
    if index !=0:
        newFold=foldsX[0]
        for i in range(1, K_FOLDS):
            if i != index:
                newFold = np.concatenate((newFold, foldsX[i]), axis=0)
    else:
        newFold=foldsX[1]
        for i in range(2, K_FOLDS):
            newFold=np.concatenate((newFold,foldsX[i]),axis=0)
    return newFold

def folds(X_train, Y_train, k):    #split folds
    testScores=[]
    newY=np.split(Y_train, K_FOLDS)
    newX=np.split(X_train, K_FOLDS)
    for i in range(0,5):
        classifier = NN.NearestNeighbor()
        classifier.train(conc(newX,i), conc(newY,i))
        Yval_predict = classifier.predict(newX[i], k, 'L1')
        num_correct = np.sum(Yval_predict == newY[i])
        num_test=len(Y_train)
        accuracy = float(num_correct) / len(newX[i])
        print 'Got %d / %d correct => accuracy: %f' % (num_correct, len(newX[i]), accuracy)
        testScores.append(accuracy)
        mean=np.mean(np.asarray(testScores))
        return mean



#code
# Load the raw CIFAR-10 data.
cifar10_dir = '/home/cse/Downloads/Untitled/cifar-10-batches-py'

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

num_training = 50000
# Subsample the data for more efficient code execution in this exercise
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 100
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

#k=int(math.sqrt(num_training/num_test))     #equation for rule of thumb k

means=[]
ks =[1,2,3,4,5,7,8,9,10,11,12,13,14,15]
for i in range(len(ks)):
        ahmed=folds(X_train, y_train, ks[i])
	means.append(ahmed)
	print "The mean accuracy for k = ", ks[i] , "is: " ,ahmeds



