# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:03:13 2019

@author: Deng
"""

import numpy as np
import copy
from sklearn.decomposition import PCA 


def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

# load feature
gf = np.load('./data/gf_ori.npy')

# tracklet name with ID
track_name = []
f = open('./data/test_track_id.txt', 'r')
for k, line in enumerate(f):
    temp = list(map(int, line.split(' ')[:-1]))
    track_name.append(list(map(lambda x: x-1, temp)))
f.close

# re-organize gallery feature
T=6 #6
for i in range(len(track_name)):
    indice = track_name[i]
    for j in range(0, len(indice), T):
        if (j+T)>len(indice):
            ind = indice[j:]
        else:
            ind = indice[j:j+T]
        gf_temp = np.mean(gf[ind], axis=0, keepdims=True) 
        gf[ind] = gf_temp

# feature norm
g_n = np.linalg.norm(gf, axis=1, keepdims=True)
gf = gf / g_n

'''
# PCA for feature
pca=PCA(n_components=1024)
gf_new = pca.fit_transform(gf)
qf_new = pca.transform(qf)
'''
np.save('./data/gf_multi.npy', gf)
