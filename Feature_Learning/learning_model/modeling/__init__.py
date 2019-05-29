# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline_de import Baseline_de

def build_model(cfg, num_classes):

    if cfg.MODEL.NAME == 'densenet121':
        model = Baseline_de(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH)
        
    return model
