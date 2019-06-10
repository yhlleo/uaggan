#! -*- coding: utf-8 -*-
# Author: Yahui Liu <yahui.liu@unitn.it>

'''
  An implement of the UAGGAN model.
  
  Unsupervised Attention-guided Image-to-Image Translation, NIPS 2018.
    https://arxiv.org/pdf/1806.02311.pdf

  Other references: 
  GANimation: Anatomically-aware Facial Animation from a Single Image.
    https://arxiv.org/pdf/1807.09251.pdf
'''

import torch
import torch.nn as nn
from .networks import init_net, get_norm_layer

class Basicblock(nn.Module):
    '''A simple version of residual block.'''
    def __init__(self, in_feat, kernel_size=3, stride=1, padding=1, norm='instance'):
        super(Basicblock, self).__init__()

        norm_layer = get_norm_layer(norm)
        residual = [nn.Conv2d(in_feat, in_feat, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    norm_layer(in_feat),
                    nn.ReLU(True),
                    nn.Conv2d(in_feat, in_feat, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    norm_layer(in_feat)]
        self.residual = nn.Sequential(*residual)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        return self.relu(x + self.residual(x))

class Bottleneck(nn.Module):
    def __init__(self, in_feat, out_feat, depth_bottleneck, stride=1, norm='instance'):
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
                    nn.ReLU(True),
                    nn.Conv2d(depth_bottleneck, depth_bottleneck, kernel_size=3, stride=stride, padding=1, bias=False),
                    norm_layer(depth_bottleneck),
                    nn.ReLU(True),
                    nn.Conv2d(depth_bottleneck, out_feat, kernel_size=1, stride=1, bias=False),
                    norm_layer(out_feat)]
        self.residual = nn.Sequential(*residual)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        preact = self.preact(x)
        if self.in_equal_out:
            shortcut = self.shortcut(x)
        else:
            shortcut = self.shortcut(preact)
        return self.relu(shortcut + self.residual(x))

class ResNetGenerator_Att(nn.Module):
    '''ResNet-based generator for attention mask prediction.'''
    def __init__(self, in_nc, ngf, norm='instance', residual_mode='basic'):
        super(ResNetGenerator_Att, self).__init__()
        assert residual_mode in ['bottleneck', 'basic']

        norm_layer = get_norm_layer(norm)
        encoder = [nn.Conv2d(in_nc, ngf, kernel_size=7, stride=2, padding=3, bias=False),
                   norm_layer(ngf),
                   nn.ReLU(True),
                   nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
                   norm_layer(ngf*2),
                   nn.ReLU(True)]

        if residual_mode == 'bottleneck':
            encoder += [Bottleneck(ngf*2, ngf*2, ngf*2, norm=norm)]
        else:
            encoder += [Basicblock(ngf*2, norm=norm)]
        self.encoder = nn.Sequential(*encoder)

        self.decoder1 = nn.Sequential(nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
                                      norm_layer(ngf*2),
                                      nn.ReLU(True))

        self.decoder2 = nn.Sequential(nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
                                      norm_layer(ngf),
                                      nn.ReLU(True),
                                      nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=3, bias=False),
                                      nn.Sigmoid())
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        encoder = self.encoder(x)
        decoder1 = self.decoder1(self.up2(encoder))
        decoder2 = self.decoder2(self.up2(decoder1))
        return decoder2

class ResNetGenerator_Img(nn.Module):
    '''ResNet-based generator for target generation.'''
    def __init__(self, in_nc, out_nc, ngf, num_blocks=9, norm='instance', residual_mode='basic'):
        super(ResNetGenerator_Img, self).__init__()
        assert residual_mode in ['bottleneck', 'basic']

        norm_layer = get_norm_layer(norm)
        model = [nn.Conv2d(in_nc, ngf, kernel_size=7, stride=1, padding=3, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(True),
                 nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
                 norm_layer(ngf*2),
                 nn.ReLU(True),
                 nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=False),
                 norm_layer(ngf*4),
                 nn.ReLU(True)]

        for i in range(num_blocks):
            if residual_mode == 'bottleneck':
                model += [Bottleneck(ngf*4, ngf*4, ngf, norm=norm)]
            else:
                model += [Basicblock(ngf*4, norm=norm)]

        model += [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2,
                                     padding=1, bias=False),
                  norm_layer(ngf*2),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2,
                                     padding=1, bias=False),
                  norm_layer(ngf),
                  nn.ReLU(True),
                  nn.Conv2d(ngf, out_nc, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResNetGenerator_v2(nn.Module):
    '''ResNet-based generator for target generation.'''
    def __init__(self, in_nc, out_nc, ngf, num_blocks=9, norm='instance', residual_mode='basic'):
        super(ResNetGenerator_v2, self).__init__()
        assert residual_mode in ['bottleneck', 'basic']

        norm_layer = get_norm_layer(norm)
        model = [nn.Conv2d(in_nc, ngf, kernel_size=7, stride=1, padding=3, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(True),
                 nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
                 norm_layer(ngf*2),
                 nn.ReLU(True),
                 nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=False),
                 norm_layer(ngf*4),
                 nn.ReLU(True)]

        for i in range(num_blocks):
            if residual_mode == 'bottleneck':
                model += [Bottleneck(ngf*4, ngf*4, ngf, norm=norm)]
            else:
                model += [Basicblock(ngf*4, norm=norm)]

        model += [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2,
                                     padding=1, bias=False),
                  norm_layer(ngf*2),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2,
                                     padding=1, bias=False),
                  norm_layer(ngf),
                  nn.ReLU(True)]
        self.model = nn.Sequential(*model)

        self.img = nn.Sequential(nn.Conv2d(ngf, out_nc, kernel_size=7, stride=1, padding=3, bias=False),
                                 nn.Tanh())

        self.att = nn.Sequential(nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=3, bias=False),
                                 nn.Sigmoid())

    def forward(self, x):
        features = self.model(x)
        return self.img(features), self.att(features)


class Discriminator(nn.Module):
    '''Discriminator'''
    def __init__(self, in_nc, ndf, n_layers=3, norm='instance', transition_rate=0.1):
        super(Discriminator, self).__init__()

        self.transition_rate = transition_rate
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
        #x_ = x*(mask>self.transition_rate).float()
        return self.model(x)

def define_net_att(in_nc, 
                   ngf, 
                   norm='instance', 
                   init_type='normal', 
                   init_gain=0.02, 
                   gpu_ids=[]):
    net = ResNetGenerator_Att(in_nc, ngf, norm=norm)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_net_img(in_nc, 
                   out_nc, 
                   ngf, 
                   num_blocks=9, 
                   norm='instance', 
                   init_type='normal', 
                   init_gain=0.02, 
                   gpu_ids=[]):
    net = ResNetGenerator_Img(in_nc, out_nc, ngf, num_blocks=num_blocks, norm=norm)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_net_faster(in_nc, 
                      out_nc, 
                      ngf, 
                      num_blocks=9, 
                      norm='instance', 
                      init_type='normal', 
                      init_gain=0.02, 
                      gpu_ids=[]):
    net = ResNetGenerator_v2(in_nc, out_nc, ngf, num_blocks=num_blocks, norm=norm)
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
