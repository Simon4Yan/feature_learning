# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:03:13 2019

@author: PAMI Deng
"""

import numpy as np
import copy


# frame inforamtion
frameQuery = np.load('./data/frameQuery.npy')
frameTracklets  = np.load('./data/frameGallery.npy')
frameGallery = np.zeros([18290])

# tracklet name with ID
g_name = []
f = open('./test_track_id.txt', 'r')
for k, line in enumerate(f):
    temp = list(map(int, line.split(' ')[:-1]))
    g_name.append(list(map(lambda x: x-1, temp)))
    for indexTemp, idTemp in enumerate(temp):
        frameGallery[idTemp-1] = frameTracklets[k]
f.close

# gallery labels and query labels
q_label = np.ones([1052])*1000
g_label = np.ones([18290])*1000

# camera inforamation
q_cams = np.load('./data/q_cams.npy')
g_cams = np.load('./data/gallery_camera_inf.npy')

# open results of S02
f = open('./data/s02_query_gallery.txt', 'r')
for k, line in enumerate(f):
    temp = line.split(' : ')
    temp0 = list(map(int, temp[0].split(' ')))
    q_label[list(map(lambda x: x-1, temp0))] = int(k)
    
    temp1 = list(map(int, temp[1].split(' ')))
    temp2 = list(map(lambda x: x-1, temp1))
    for i in range(len(temp2)): 
        g_label[np.array(g_name)[temp2[i]]] = int(k)
f.close()

q_index = np.nonzero(q_label!=1000)[0]
g_index = np.nonzero(g_label!=1000)[0]

#-------------------------------------- query expansion version 1 -------------------------#
# load feature
gf = np.load('./data/gf_multi.npy')
qf = np.load('./data/qf_multi.npy')
#dist = np.load('./data/distmat.npy')

dist = np.dot(qf, np.transpose(gf))
dist = 2. - 2 * dist  # change the cosine similarity metric to euclidean similarity metric
d_max = 10.0

# time filter
frameGap = 190
for i in range(len(q_index)):
    key = q_index[i]
    indice = abs(frameGallery - frameQuery[key])> frameGap
    if q_cams[key]==6:
       indice = indice | (abs(frameGallery - frameQuery[key])> 60) & (g_cams == 9)
    elif q_cams[key] == 9:
       indice = indice | (abs(frameGallery - frameQuery[key])> 50) & (g_cams == 6)
    indice = np.nonzero(indice)[0]
    dist[key][indice] = d_max

# camera filter for S02 and S05
for i in range(len(dist)):
    indice =  np.nonzero((g_cams == q_cams[i]))[0]
    dist[i][indice] = d_max

# scene filter for S02
g_indice_scene = np.nonzero(g_label==1000)[0] #S05 indice
for i in range(len(q_index)):
    key = q_index[i]
    dist[key][g_indice_scene] = d_max

# scene filter for S05
g_indice_scene = np.nonzero(g_label!=1000)[0] #S02 indice
s05_indice =  np.nonzero(q_label==1000)[0]
for i in range(len(s05_indice)):
    key = s05_indice[i]
    dist[key][g_indice_scene] = d_max
        
#---------------------------- query expansion  ----------------------------#
qf_new = []
T = 9
num = 1
d_max = 10.0

for t in range(num):
    qf_new = []
    for i in range(len(dist)):
        indice = np.argsort(dist[i])[:T]
        temp = np.concatenate((qf[i][np.newaxis, :], gf[indice]), axis=0)
        #temp = gf[indice]
        qf_new.append(np.mean(temp, axis=0, keepdims=True)) 
    qf = np.squeeze(np.array(qf_new))
    # feature norm
    q_n = np.linalg.norm(qf, axis=1, keepdims=True)
    qf = qf / q_n
    
    dist = np.dot(qf, np.transpose(gf))
    dist = 2. - 2 * dist  # change the cosine similarity metric to euclidean similarity metric
    #np.save('./data/gf.npy', gf)
    np.save('./data/qf_new_%d.npy' % t, qf)

    # time filter
    frameGap = 190
    for i in range(len(q_index)):
        key = q_index[i]
        indice = abs(frameGallery - frameQuery[key])> frameGap
        if q_cams[key]==6:
           indice = indice | (abs(frameGallery - frameQuery[key])> 60) & (g_cams == 9)
        elif q_cams[key] == 9:
           indice = indice | (abs(frameGallery - frameQuery[key])> 50) & (g_cams == 6)
        indice = np.nonzero(indice)[0]
        dist[key][indice] = d_max
    
    # camera filter for S02 and S05
    for i in range(len(dist)):
        indice =  np.nonzero((g_cams == q_cams[i]))[0]
        dist[i][indice] = d_max
    
    # scene filter for S02
    g_indice_scene = np.nonzero(g_label==1000)[0] #S05 indice
    for i in range(len(q_index)):
        key = q_index[i]
        dist[key][g_indice_scene] = d_max
    
    # scene filter for S05
    g_indice_scene = np.nonzero(g_label!=1000)[0] #S02 indice
    s05_indice =  np.nonzero(q_label==1000)[0]
    for i in range(len(s05_indice)):
        key = s05_indice[i]
        dist[key][g_indice_scene] = d_max

'''
#-------------------------------------- query expansion version 2 -------------------------#
# load feature
gf = np.load('./data/gf_multi.npy')
qf = np.load('./data/qf_multi.npy')

f = open('./results/track2_time_all_order.txt', 'r')
dist = []

for line in f:
    temp0 = list(map(int, line.strip(' \n').split(' ')))
    temp = list(map(lambda x: x-1, temp0))
    dist.append(temp)
    
dist = np.array(dist)
dist = np.load('./data/distmat.npy')
qf_new = []
T = 10
weights = 1

for i in range(len(dist)):
    indice = np.argsort(dist[i])[:T]
    #temp = gf[indice]
    temp = np.mean(gf[indice], axis=0, keepdims=True)
    temp = np.concatenate((qf[i][np.newaxis, :], temp), axis=0)
    qf_new.append(np.mean(temp, axis=0, keepdims=True)) 
qf = np.squeeze(np.array(qf_new))
# feature norm
q_n = np.linalg.norm(qf, axis=1, keepdims=True)
qf = qf / q_n

dist = np.dot(qf, np.transpose(gf))
dist = 2. - 2 * dist  # change the cosine similarity metric to euclidean similarity metric
#np.save('./data/gf.npy', gf)
np.save('./data/qf_new_0.npy', qf)
'''
