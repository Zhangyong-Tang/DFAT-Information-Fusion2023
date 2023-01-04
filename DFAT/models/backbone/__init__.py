# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from DFAT.models.backbone.alexnet import alexnetlegacy, alexnet
from DFAT.models.backbone.mobile_v2 import mobilenetv2
from DFAT.models.backbone.resnet_atrous import resnet18, resnet34, resnet50
from DFAT.models.backbone.resnet import resnet18_, resnet34_, resnet50_


BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
              'resnet18_': resnet18_,
              'resnet34_': resnet34_,
              'resnet50_': resnet50_,
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
