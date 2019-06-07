#! -*- coding: utf-8 -*-
# Author: Yahui Liu <yahui.liu@unitn.it>

'''
  An implement of the UAGGAN model.
  
  Paper: Unsupervised Attention-guided Image-to-Image Translation, NIPS 2018.
         https://arxiv.org/pdf/1806.02311.pdf
'''

import torch
import torch.nn as nn
from .networks import init_net, get_norm_layer

class Basicblock(nn.Module):
    '''A simple version of residual block.'''
    def __init__(self, in_feat, out_feat=0, depth_bottleneck=0, kernel_size=3, stride=1, padding=1, norm='instance'):
        super(Basicblock, self).__init__()

        norm_layer = get_norm_layer(norm)
        residual = [nn.Conv2d(in_feat, in_feat, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    norm_layer(in_feat),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_feat, in_feat, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    norm_layer(in_feat),
                    nn.ReLU(inplace=True)]
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        return x + self.residual(x)

class Bottleneck(nn.Module):
    '''ResNet v2 residual block.'''
    def __init__(self, in_feat, out_feat, depth_bottleneck, kernel_size=1, stride=1, padding=0, norm='instance'):
        super(Bottleneck, self).__init__()

        norm_layer = get_norm_layer(norm)
        self.in_equal_out = in_feat == out_feat
        
        self.preact = nn.Sequential(norm_layer(in_feat),
                                    nn.ReLU(inplace=True))

        if self.in_equal_out:
            self.shortcut = nn.MaxPool2d(1, stride=stride)
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False))

        residual = [nn.Conv2d(in_feat, depth_bottleneck, kernel_size=1, stride=1, bias=False),
                    norm_layer(depth_bottleneck),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(depth_bottleneck, depth_bottleneck, kernel_size=3, stride=stride, padding=1, bias=False),
                    norm_layer(depth_bottleneck),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(depth_bottleneck, out_feat, kernel_size=1, stride=1, bias=False),
                    norm_layer(out_feat)]
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        preact = self.preact(x)
        if self.in_equal_out:
            shortcut = self.shortcut(x)
        else:
            shortcut = self.shortcut(preact)
        return shortcut + self.residual(x)

class ResNetGenerator_Att(nn.Module):
    '''ResNet-based generator for attention mask prediction.'''
    def __init__(self, in_nc, ngf, norm='instance', block_mode='basic'):
        super(ResNetGenerator_Att, self).__init__()
        assert block_mode in ['bottleneck', 'basic']

        norm_layer = get_norm_layer(norm)
        model = [nn.Conv2d(in_nc, ngf, kernel_size=7, stride=2, padding=3, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(inplace=True)]

        if block_mode == 'bottleneck':
            model += [Bottleneck(ngf*2, ngf*2, ngf*2, norm=norm)]
        else:
            model += [Basicblock(ngf*2, norm=norm)]

        model += [nn.ConvTranspose2d(ngf*2, ngf*2, kernel_size=3, stride=2,
                                     padding=1, output_padding=1, bias=False),
                  norm_layer(ngf*2),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
                  norm_layer(ngf*2),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(ngf*2, ngf*2, kernel_size=3, stride=2,
                                     padding=1, output_padding=1, bias=False),
                  norm_layer(ngf*2),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
                  norm_layer(ngf),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.Sigmoid()]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class ResNetGenerator_Img(nn.Module):
    '''ResNet-based generator for target generation.'''
    def __init__(self, in_nc, out_nc, ngf, num_blocks=9, norm='instance', block_mode='basic'):
        super(ResNetGenerator_Img, self).__init__()
        assert block_mode in ['bottleneck', 'basic']

        norm_layer = get_norm_layer(norm)
        model = [nn.Conv2d(in_nc, ngf, kernel_size=7, stride=1, padding=3, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
                 norm_layer(ngf*2),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=False),
                 norm_layer(ngf*4),
                 nn.ReLU(inplace=True)]

        baseblock = Bottleneck if block_mode == 'bottleneck' else Basicblock
        for i in range(num_blocks):
            model += [baseblock(ngf*4, ngf*4, ngf, norm=norm)]

        model += [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=False),
                  norm_layer(ngf*2),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=False),
                  norm_layer(ngf),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(ngf, out_nc, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    '''Discriminator'''
    def __init__(self, in_nc, ndf, n_layers=3, norm='instance'):
        super(Discriminator, self).__init__()

        norm_layer = get_norm_layer(norm)
        model = [nn.Conv2d(in_nc, ndf, kernel_size=4, stride=2, padding=1, bias=False), 
                 norm_layer(ndf),
                 nn.LeakyReLU(0.2, True)]

        cur_in, cur_out = ndf, ndf
        for i in range(n_layers):
            cur_in = cur_out
            cur_out =  ndf * min(2**i, 8)
            model += [nn.Conv2d(cur_in, cur_out, kernel_size=4, stride=2, padding=1, bias=False), 
                      norm_layer(cur_out),
                      nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(cur_out, 1, kernel_size=4, stride=1, padding=1, bias=False)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def define_net_att(in_nc, 
                   ngf, 
                   norm='instance', 
                   block_mode='basic',
                   init_type='normal', 
                   init_gain=0.02, 
                   gpu_ids=[]):
    net = ResNetGenerator_Att(in_nc, ngf, norm=norm, block_mode=block_mode)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_net_img(in_nc, 
                   out_nc, 
                   ngf, 
                   num_blocks=9, 
                   norm='instance', 
                   block_mode='basic',
                   init_type='normal', 
                   init_gain=0.02, 
                   gpu_ids=[]):
    net = ResNetGenerator_Img(in_nc, out_nc, ngf, num_blocks=num_blocks, norm=norm, block_mode=block_mode)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_net_dis(in_nc, 
                   ndf, 
                   n_layers=3, 
                   norm='instance', 
                   init_type='normal', 
                   init_gain=0.02, 
                   gpu_ids=[]):
    net = Discriminator(in_nc, ndf, n_layers, norm=norm)
    return init_net(net, init_type, init_gain, gpu_ids)
