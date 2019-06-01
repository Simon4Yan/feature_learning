# encoding: utf-8
"""
@author:  weijian
@contact: dengwj16@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
from data.datasets.eval_reid import eval_func
import logging
import numpy as np

from utils.re_ranking import re_ranking

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def extract_feature(model, dataloaders, num_query):
    features = []
    count = 0
    img_path = []

    for data in dataloaders:
        img, _, _ = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 1024).zero_().cuda() # 2048 is pool5 of resnet
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = img.cuda()
            outputs = model(input_img) 
            f = outputs.float()
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features.append(ff)
    features = torch.cat(features, 0)   
    
    # query
    qf = features[:num_query]
    # gallery
    gf = features[num_query:]
    return qf, gf

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
    model = model.cuda()
    model = model.eval()
    
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")
    with torch.no_grad():
        qf, gf = extract_feature(model, val_loader, num_query)
    
    # save feature
    np.save('../data/feature_expansion/qf_ori', qf.cpu().numpy())
    np.save('../data/feature_expansion/gf_ori', gf.cpu().numpy())
    '''
    q_g_dist = np.dot(qf.cpu().numpy(), np.transpose(gf.cpu().numpy()))
    q_q_dist = np.dot(qf.cpu().numpy(), np.transpose(qf.cpu().numpy()))
    g_g_dist = np.dot(gf.cpu().numpy(), np.transpose(gf.cpu().numpy()))
      
    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    '''
    
if __name__ == '__main__':
    main()
