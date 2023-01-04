from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from DFAT.core.config import cfg
EPSILON = 1e-10
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.bn(out)
        # if self.is_last is False:
            # out = F.normalize(out)
        # out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


class RFN(nn.Module):
    def __init__(self, input=256, kernel=3):
        super(RFN, self).__init__()
        if kernel==3:
            stride=1
        else:
            stride=1
        self.fusion = ConvLayer(input * 2, input, kernel_size=3, stride=1, is_last=True)
        self.rgb = ConvLayer(input, input, kernel_size=1, stride=1)
        self.ir = ConvLayer(input, input, kernel_size=1, stride=1)
        block = []
        block += [ConvLayer(2*input, input, 1, 1),
                  ConvLayer(input, input, 3, 1),
                  ConvLayer(input, input, 3, 1, is_last=True)]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_rgb, x_ir):
        # initial fusion - conv
        # print('conv')
        f_cat = torch.cat([x_rgb, x_ir], 1)
        f_init = self.fusion(f_cat)

        out_rgb = self.rgb(x_rgb)
        out_ir = self.ir(x_ir)
        out = torch.cat([out_rgb, out_ir], 1)
        out = self.bottelblock(out)
        out = f_init + out
        return out

# spatial attention
def spatial_attention(tensor, spatial_type='sum', act=True):
	spatial = None
	if spatial_type == 'mean':
		spatial = tensor.mean(dim=1, keepdim=True)
	elif spatial_type == 'sum':
		spatial = tensor.sum(dim=1, keepdim=True)
	elif spatial_type == 'max':
		spatial, _ = tensor.max(dim=1, keepdim=True)
	if act is True:
		spatial = F.sigmoid(spatial)
	return spatial

def fusion_spatial(f_rgb, f_ir, spatial_type):
	# type = 'mean'
	act_f = True
	shape = f_rgb.size()
	feature_rgb = f_rgb
	feature_ir = f_ir

	# spatial attention
	spatial1 = spatial_attention(feature_rgb, spatial_type=spatial_type, act=act_f)
	spatial2 = spatial_attention(feature_ir, spatial_type=spatial_type, act=act_f)
	# get weight map, soft-max
	spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
	spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
	# fusion
	spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
	spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
	tensor_f = spatial_w1 * feature_rgb + spatial_w2 * feature_ir
	return tensor_f

class mRFN(nn.Module):
    def __init__(self, input=256, kernel=3):
        super(mRFN, self).__init__()
        self.num = len(cfg.RPN.KWARGS.in_channels)
        self.spatial_type = 'mean'
        for i in range(self.num):
            name = 'RFN' + str(i)
            self.add_module(name, RFN(input, kernel))

    def forward(self, x_rgb, x_ir):
        out = []
        for i in range(len(x_rgb)):
            name = 'RFN' + str(i)
            rfn = self.__getattr__(name)
            out.append(rfn(x_rgb[i], x_ir[i]))
        return out

    def spatial_model(self, en_rgb, en_ir):
        sf_0 = fusion_spatial(en_rgb[0], en_ir[0], self.spatial_type)
        sf_1 = fusion_spatial(en_rgb[1], en_ir[1], self.spatial_type)
        sf_2 = fusion_spatial(en_rgb[2], en_ir[2], self.spatial_type)
        return [sf_0, sf_1, sf_2]