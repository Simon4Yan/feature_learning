# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:03:13 2019

@author: Deng
"""

import numpy as np


# frame inforamtion
frameQuery = np.load('./stamp/frameQuery.npy')
frameTracklets  = np.load('./stamp/frameGallery.npy')
frameGallery = np.zeros([18290])

# tracklet name with ID
g_name = []
f = open('./data/test_track_id.txt', 'r')
for k, line in enumerate(f):
    temp = list(map(int, line.split(' ')[:-1]))
    g_name.append(list(map(lambda x: x-1, temp)))
    for indexTemp, idTemp in enumerate(temp):
        frameGallery[idTemp-1] = frameTracklets[k]
f.close

# camera inforamation
q_cams = np.load('./stamp/q_cams.npy')
g_cams = np.load('./stamp/gallery_camera_inf.npy')

# location information
g_index_s05 = np.load('./stamp/g_index_s05.npy')
q_index_s05 = np.load('./stamp/q_index_s05.npy')

q_index = np.load('./stamp/q_index_s02.npy')
g_index = np.load('./stamp/g_index_s02.npy')

#---------------   query expansion with location and time info --------------#
# load feature
gf = np.load('./data/gf_multi.npy')
qf = np.load('./data/qf_multi.npy')

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
g_indice_scene = g_index_s05 #S05 indice
for i in range(len(q_index)):
    key = q_index[i]
    dist[key][g_indice_scene] = d_max

# scene filter for S05
g_indice_scene = g_index #S02 indice
s05_indice =  q_index_s05
for i in range(len(s05_indice)):
    key = s05_indice[i]
    dist[key][g_indice_scene] = d_max
        
#---------------------------- query expansion  ----------------------------#
qf_new = []
T = 9

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
np.save('./data/qf_new.npy', qf)
