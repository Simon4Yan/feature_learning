# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:03:13 2019

@author: Deng
"""

import numpy as np


#--------------------------- query expansion -----------------#
# load feature
gf = np.load('./data/gf_multi.npy')
qf = np.load('./data/qf_ori.npy')

# load indice
qf_new = np.load('./data/qf_ori.npy')
query_index = []
f = open('./data/query_new.txt', 'r')
for k, line in enumerate(f):
    temp = int(line[0:6]) - 1
    query_index.append(temp)
    qf[k] = qf_new[temp]
f.close()

# feature norm
q_n = np.linalg.norm(qf, axis=1, keepdims=True)
qf = qf / q_n

dist = np.dot(qf, np.transpose(gf))
dist = 2. - 2 * dist  # change the cosine similarity metric to euclidean similarity metric
       
qf_new = []
T = 9
num = 1
d_max = 10.0

for t in range(num):
    qf_new = []
    for i in range(len(dist)):
        indice = np.argsort(dist[i])[:T]
        temp = np.concatenate((qf[i][np.newaxis, :], gf[indice]), axis=0)
        qf_new.append(np.mean(temp, axis=0, keepdims=True))
        
    qf = np.squeeze(np.array(qf_new))
    # feature norm
    q_n = np.linalg.norm(qf, axis=1, keepdims=True)
    qf = qf / q_n
    
    dist = np.dot(qf, np.transpose(gf))
    dist = 2. - 2 * dist  # change the cosine similarity metric to euclidean similarity metric
    np.save('./data/qf_multi_%d.npy' % t, qf)   
    