from __future__ import print_function
import numpy as np
from math import sqrt
import scipy as sp
import os, sys, copy, pickle
from scipy import io as sio
from sklearn.metrics import roc_auc_score
from datetime import datetime
import pickle as pkl
import random
random.seed(0)

def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def normalizeWH(W, H):
    print("Normalizing W and H")
    K = H.shape[0]
    D = np.eye(K)
    for k in range(K):
        sum_Hk = sqrt(sum([h**2 for h in H[k]]))
        D[k, k] = sum_Hk
    pnk_W = np.dot(W, D)

    D = np.eye(K)
    for k in range(K):
        sum_Wk = sqrt(sum([w**2 for w in W[:,k]]))
        D[k, k] = sum_Wk
    pnk_H = np.dot(D, H)

    return pnk_W, pnk_H


def read_ratings_movielens(fn):
    if type(fn) == type({}):
        ratings = fn
        num_ratings = ratings['ratings'].shape[1]
        print("Total number of ratings: %d" % num_ratings)
        for i in range(num_ratings):
            user, item, rating = ratings['users'][0, i], ratings['items'][0, i], ratings['ratings'][0, i]
            print("%d, %d, %d" % (user, item, rating))
            if i > 10:
                break
    elif '.mat' in fn:
        ratings = sio.loadmat(fn)
        num_ratings = ratings['ratings'].shape[1]
        print("Total number of ratings: %d" % num_ratings)
        for i in range(num_ratings):
            user, item, rating = ratings['users'][0, i], ratings['items'][0, i], ratings['ratings'][0, i]
            print("%d, %d, %d" % (user, item, rating))
            if i > 10:
                break
    elif '.dat' in fn:
        ratings = open(fn, 'r')
        cnt = 0
        for line in ratings:
            print(line)
            cnt +=1
            if cnt > 10:
                break

import pdb

def load_lp_eval_data(num_nodes, data):
    label = np.zeros(num_nodes)

    for node in data:
        label[node] = data[node]

    valid_idxs = np.array(list(data.keys()))
    return label[valid_idxs]

def split_data(num_nodes):
    idxs = list(range(num_nodes))
    random.shuffle(idxs)

    train_idxs = idxs[:int(len(idxs) * 0.8)]
    val_idxs = idxs[int(len(idxs) * 0.8): int(len(idxs) * 0.9)]
    test_idxs = idxs[int(len(idxs) * 0.9):]
    # train_idxs = idxs[:int(len(idxs) * 0.8)]
    # val_idxs = idxs[int(len(idxs) * 0.8): int(len(idxs) * 0.9)]
    # test_idxs = idxs[int(len(idxs) * 0.9):]

    return train_idxs, val_idxs, test_idxs