# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:03:13 2019

@author: Deng
"""

import numpy as np

#--------------------------- load feature -----------------#
# load feature
gf_1 = np.load('./data/gf_1.npy')
gf_1_n = np.linalg.norm(gf_1, axis=1, keepdims=True)
gf_1 = gf_1 / gf_1_n

qf_1 = np.load('./data/qf_1.npy')
qf_1_n = np.linalg.norm(qf_1, axis=1, keepdims=True)
qf_1 = qf_1 / qf_1_n

# model 2
gf_2 = np.load('./data/gf_2.npy')
gf_2_n = np.linalg.norm(gf_2, axis=1, keepdims=True)
gf_2 = gf_2 / gf_2_n

qf_2 = np.load('./data/qf_2.npy')
qf_2_n = np.linalg.norm(qf_2, axis=1, keepdims=True)
qf_2 = qf_2 / qf_2_n

# model 3
gf_3 = np.load('./data/gf_3.npy')
gf_3_n = np.linalg.norm(gf_3, axis=1, keepdims=True)
gf_3 = gf_3 / gf_3_n

qf_3 = np.load('./data/qf_3.npy')
qf_3_n = np.linalg.norm(qf_3, axis=1, keepdims=True)
qf_3 = qf_3 / qf_3_n

#---------------------------  feature concat -----------------#
qf_ori = np.concatenate((qf_1, qf_2, qf_3), axis=1) /np.sqrt(3)
gf_ori = np.concatenate((gf_1, gf_2, gf_3), axis=1) /np.sqrt(3)

#---------------------------  save feature -----------------#
np.save('./data/qf_ori.npy', qf_ori)   
np.save('./data/gf_ori.npy', gf_ori) 
      