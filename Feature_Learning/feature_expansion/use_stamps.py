# -*- coding: utf-8 -*-

"""
Created on Sat Apr 27 15:03:13 2019

@author: Lv
"""

import numpy as np
import copy


# frame inforamtion
frameQuery = np.load('./stamp/frameQuery.npy')
frameTracklets  = np.load('./stamp/frameGallery.npy')
frameGallery = np.zeros([18290]) #18290 is the num of gallery images

# tracklet name with ID
g_name = []
f = open('./data/test_track_id.txt', 'r')
for k, line in enumerate(f):
    temp = list(map(int, line.split(' ')[:-1]))
    if len(temp)==1:
        continue
    g_name.append(list(map(lambda x: x-1, temp)))
    for indexTemp, idTemp in enumerate(temp):
        frameGallery[idTemp-1] = frameTracklets[k]
f.close

# camera inforamation
q_cams = np.load('./stamp/q_cams.npy')
g_cams = np.load('./stamp/gallery_camera_inf.npy')

# distance matrix
qf = np.load('./data/qf_new.npy')
gf = np.load('./data/gf_multi.npy')
distori = np.dot(qf, np.transpose(gf))
distori = 2. - 2 * distori  # change the cosine similarity metric to euclidean similarity metric

distmat = copy.deepcopy(distori)

distance1 = np.zeros(36)
distance1[27-1] = 0
distance1[28-1] = 28
distance1[29-1] = 46
distance1[33-1] = 60
distance1[34-1] = 76
distance1[35-1] = 84
distance1[36-1] = 90

distance2 = np.zeros(36)
distance2[27-1] = 94
distance2[28-1] = 42
distance2[29-1] = 36
distance2[33-1] = 25
distance2[34-1] = 21
distance2[35-1] = 13
distance2[36-1] = 0

framegaps = np.ones([36,36])*400
framegaps[26,:]= 460
framegaps[27,:]= 180
framegaps[28,:]= 240
framegaps[32,:]= 270
framegaps[33,:]= 170
framegaps[34,:]= 130
framegaps[35,:]= 120

orisQuery = np.load('./stamp/orisQuery.npy')
orisGallery = np.load('./stamp/orisGallery.npy')

def func_stamps(distmat,  q_camids, g_camids, frameQuery, frameGallery, framegaps,
                                   distance1, distance2, orisQuerys, orisGallerys, max_rank=100):
    
    num_q, num_g = distmat.shape
    newdist = np.zeros([num_q, num_g])
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)

    for q_idx in range(num_q):
        q_camid = q_camids[q_idx]
        q_frame = frameQuery[q_idx]
        oriQuery = orisQuerys[int(q_idx)]
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = g_camids[order] == q_camid

        if q_camid < 10 :
            if q_camid == 6:
                remove = remove | ((abs(frameGallery[order] - q_frame) > 60) & (g_camids[order] == 9))
            elif q_camid == 9:
                remove = remove | ((abs(frameGallery[order] - q_frame) > 50) & (g_camids[order] == 6))
        else:
            cameras = [27, 28, 29, 33, 34, 35, 36]
            distQG = q_frame - frameGallery[order]
            for camera in cameras:
                if camera == q_camid:
                    continue
                cameraGallery = camera
                if oriQuery == 0:
                    center = (distance1[q_camid - 1] - distance1[cameraGallery - 1]) * 10
                else:
                    center = (distance2[q_camid - 1] - distance2[cameraGallery - 1]) * 10
                remove = remove | ((abs(distQG - center) > framegaps[q_camid - 1, cameraGallery - 1]) & (
                            g_camids[order] == cameraGallery))
            if oriQuery == 0:
                remove = remove | (orisGallery == 1.0)
            else:
                remove = remove | (orisGallery == 0.0)
        # filter retrieve results
        keep = np.invert(remove)
        newdist[q_idx, :np.sum(keep)] = distmat[q_idx][keep]
        distmat[q_idx][remove] = 2 # set distance to max number
   
    return distmat

# fliter retrieve results by using location and time stamps
dist = func_stamps(distmat,  q_cams, g_cams, frameQuery, frameGallery, framegaps,
                                   distance1, distance2,  orisQuery, orisGallery)

# location information
g_index_s05 = np.load('./stamp/g_index_s05.npy')
q_index_s05 = np.load('./stamp/q_index_s05.npy')

q_index = np.load('./stamp/q_index_s02.npy')
g_index = np.load('./stamp/g_index_s02.npy')

# scene filter for S02
g_indice_scene = g_index_s05 #S05 indice
for i in range(len(q_index)):
    key = q_index[i]
    dist[key][g_indice_scene] = 2

# scene filter for S05
g_indice_scene = g_index #S02 indice
s05_indice =  q_index_s05
for i in range(len(s05_indice)):
    key = s05_indice[i]
    dist[key][g_indice_scene] = 2
# save results                                   
np.save('./results/distmat_after.npy', dist)
