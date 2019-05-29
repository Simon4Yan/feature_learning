# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .VR import VR
from .VeRi import VeRi
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'VR': VR,
    'VeRi': VeRi,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
