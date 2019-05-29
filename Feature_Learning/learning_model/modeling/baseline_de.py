# encoding: utf-8
"""
@author:  weijian
@contact: dengwj16@gmail.com
"""

from torch import nn

from .backbones.torchvision_models import densenet121

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def freeze_bn(modules):            
    for m in modules:
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            
class Baseline_de(nn.Module):
    in_planes = 1024

    def __init__(self, num_classes, last_stride, model_path):
        super(Baseline_de, self).__init__()
        self.base = densenet121()
        self.base.features.transition3.pool=nn.Sequential()# stride

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_bottleneck = 1024      
        self.num_classes = num_classes
        
        add_block = []
        add_block += [nn.Linear(self.in_planes,  self.num_bottleneck)]
        add_block += [nn.BatchNorm1d(self.num_bottleneck)]
        add_block = nn.Sequential(*add_block)
        self.bottleneck = add_block
        self.bottleneck.apply(weights_init_kaiming)
               
        self.classifier = nn.Linear(self.num_bottleneck, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.gap(self.base(x))  # (b, 1024, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 1024)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat #feat or global_feat for triplet loss
        else:
            return global_feat ## feat
