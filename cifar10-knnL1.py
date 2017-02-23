
# coding: utf-8

#http://cs231n.github.io/classification/
import random
import math
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# Load the raw CIFAR-10 data.
cifar10_dir = '/home/cse/Downloads/Untitled/cifar-10-batches-py'

num_training = 10000
num_test = 10000
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)


num_training = 10000
num_test = 10000
# Subsample the data for more efficient code execution in this exercise
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

import NearestNeighbor as NN
classifier = NN.NearestNeighbor()
classifier.train(X_train, y_train)
k=int(math.sqrt(num_training/num_test))

ks =[3,4,5,7,8,9,15]

for i in range(len(ks)):
	print 'k= %d' % ks[i]
	Yval_predict = classifier.predict(X_test, ks[i], 'L1')
	num_correct = np.sum(Yval_predict == y_test)
	accuracy = float(num_correct) / num_test
	print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
