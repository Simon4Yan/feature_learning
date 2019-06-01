import numpy as np

import torch




def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    # x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    # y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    x = torch.from_numpy(query_features)
    y = torch.from_numpy(gallery_features)
    m, n = x.size(0), y.size(0)
    #
    # x = x.view(m, -1)
    # y = y.view(n, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    dist.addmm_(1, -2, x, y.t())
    return dist

featuresQuery = np.load('qf_multi.npy')
featuresGallery = np.load('gf.npy')



distmat = pairwise_distance(featuresQuery, featuresGallery)
np.save('distmat.npy',distmat)

