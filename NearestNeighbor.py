# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance

class NearestNeighbor(object):
    # http://cs231n.github.io/classification/

    def __init__(self):
        pass

    def train(self, xin, yin):
        # the nearest neighbor classifier simply remembers all the training
        # data
        self.Xtr = xin
        self.ytr = yin

    def predict(self, X, k, l='L1'):
        if l == 'L2':
            ahmed = distance.cdist(X, self.Xtr, 'euclidean')
        else:
            # cityblock is the same as manhattan distance
            ahmed = distance.cdist(X, self.Xtr, 'cityblock')
        labels = np.zeros(len(X))
        for i in range(len(labels)):
            partition = np.argpartition(ahmed[i], k)[:k]
            labels[i]=np.argmax(np.bincount(self.ytr[partition]))
        return labels
