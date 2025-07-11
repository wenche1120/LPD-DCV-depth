
from __future__ import absolute_import, division, print_function
# from pyexpat import features

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.jit import script


class WSConv2d(nn.Conv2d):
    def __init___(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv_ws(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return WSConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

@script
def _mish_jit_fwd(x): return x.mul(torch.tanh(F.softplus(x)))

@script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))

class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_jit_bwd(x, grad_output)

# Cell
def mish(x): return MishJitAutoFn.apply(x)

class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)


class upConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, norm, act, num_groups):
        super(upConvLayer, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        else:
            self.norm = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

        self.act = act
        self.scale_factor = scale_factor
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)     # pre-activation
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.conv(x)
        return x

class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1,
                    padding=0, dilation=1, bias=True, norm='BN', act='ReLU', num_groups=32):
        super(myConv, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        module = []
        if norm == 'GN':
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)
        module.append(conv(in_ch, out_ch, kernel_size=kSize, stride=stride,
                            padding=padding, dilation=dilation, groups=1, bias=bias))
        self.module = nn.Sequential(*module)
    def forward(self, x):
        out = self.module(x)
        return out


class Dilated_bottleNeck(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck, self).__init__()
        conv = conv_ws
        self.reduction1 = conv(in_feat, in_feat//2, kernel_size=1, stride = 1, bias=False, padding=0)
        self.aspp_d3 = nn.Sequential(myConv(in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=3, dilation=3,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d6 = nn.Sequential(myConv(in_feat//2 + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2 + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=6, dilation=6,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d12 = nn.Sequential(myConv(in_feat, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=12, dilation=12,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d18 = nn.Sequential(myConv(in_feat + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=18, dilation=18,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.reduction2 = myConv(((in_feat//4)*4) + (in_feat//2), in_feat//2, kSize=3, stride=1, padding=1,bias=False, norm=norm, act=act, num_groups = ((in_feat//4)*4 + (in_feat//2))//16)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3], dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6], dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12], dim=1)
        d18 = self.aspp_d18(cat3)
        out = self.reduction2(torch.cat([x, d3, d6, d12, d18], dim=1))
        return out

class Dilated_bottleNeck_lv6(nn.Module):
    def __init__(self, norm, act):
        super(Dilated_bottleNeck_lv6, self).__init__()
        conv = conv_ws
        in_feat = 2048
        in_feat = in_feat//2
        self.reduction1 = myConv(in_feat*2, in_feat, kSize=3, stride=1, padding=1, bias=False, norm=norm, act=act, num_groups=(in_feat)//16)
        self.aspp_d3 = nn.Sequential(myConv(in_feat, in_feat//2, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act, num_groups=(in_feat//2)//16),
                                    myConv(in_feat//2, in_feat//2, kSize=3, stride=1, padding=3, dilation=3, bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d6 = nn.Sequential(myConv(in_feat + in_feat//2, in_feat//2, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2 + in_feat//4)//16),
                                    myConv(in_feat//2, in_feat//2, kSize=3, stride=1, padding=6, dilation=6, bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d12 = nn.Sequential(myConv(in_feat*2, in_feat//2, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act, num_groups=(in_feat)//16),
                                    myConv(in_feat//2, in_feat//2, kSize=3, stride=1, padding=12, dilation=12, bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d18 = nn.Sequential(myConv(in_feat*2 + in_feat//2, in_feat//2, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat + in_feat//4)//16),
                                    myConv(in_feat//2, in_feat//2, kSize=3, stride=1, padding=18, dilation=18, bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.reduction2 = myConv(((in_feat//2)*4) + in_feat, in_feat*2, kSize=3, stride=1, padding=1, bias=False, norm=norm, act=act, num_groups = ((in_feat//4)*4 + (in_feat//2))//16)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3], dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6], dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12], dim=1)
        d18 = self.aspp_d18(cat3)
        out = self.reduction2(torch.cat([x, d3, d6, d12, d18], dim=1))
        return out


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.fixList = ['layer1.0', 'layer1.1', '.bn']

        for name, parameters in self.encoder.named_parameters():
            if name == 'conv1.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))

        return self.features


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with_1, concat_with_2):
        up_x = F.interpolate(x, size=[concat_with_1.size(2), concat_with_1.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with_1, concat_with_2], dim=1)

        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=512):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        norm = 'BN'
        act = 'ReLU'

        self.ASPP = Dilated_bottleNeck_lv6(norm, act)
        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        # for res50
        self.up1 = UpSampleBN(skip_input=features // 1 + 1024 + 3, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 512 + 3, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 256 + 3, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 64 + 3, output_features=features // 16)
        self.up5 = UpSampleBN(skip_input=features // 16 + 3 + 3, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)


    def forward(self, features, rgb):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[1], features[2], features[3], features[4]
        rgb_lv6, rgb_lv5, rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5]
        zero_block = torch.zeros_like(rgb_lv1)
        x_d0 = self.ASPP(x_block4)
        x_d0 = self.conv2(x_d0)

        x_d1 = self.up1(x_d0, x_block3, rgb_lv5)
        x_d2 = self.up2(x_d1, x_block2, rgb_lv4)
        x_d3 = self.up3(x_d2, x_block1, rgb_lv3)
        x_d4 = self.up4(x_d3, x_block0, rgb_lv2)

        x_d5 = self.up5(x_d4, zero_block, rgb_lv1)
        out = self.conv3(x_d5)

        return out


class Lap_decoder_lv5(nn.Module):
    def __init__(self, dimList):
        super(Lap_decoder_lv5, self).__init__()
        norm = 'BN'
        act = 'ReLU'

        kSize = 3

        self.ASPP = Dilated_bottleNeck(norm, act, dimList[3])
        self.dimList = dimList

        self.decoder1 = nn.Sequential(
            myConv(dimList[3] // 2, dimList[3] // 4, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 2) // 16),
            myConv(dimList[3] // 4, dimList[3] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 4) // 16),
            myConv(dimList[3] // 8, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 8) // 16),
            myConv(dimList[3] // 16, dimList[3] // 32, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 16) // 16),
            )

        self.decoder2_up1 = upConvLayer(dimList[3] // 2, dimList[3] // 4, 2, norm, act, (dimList[3] // 2) // 16)
        self.decoder2_reduc1 = myConv(dimList[3] // 4 + dimList[2], dimList[3] // 4 - 35, kSize=1, stride=1, padding=0, bias=False,
                                      norm=norm, act=act, num_groups=(dimList[3] // 4 + dimList[2]) // 16)  # dimList[3]//4 - 4  输出通道数-4 因为后面要拼接3+1
        self.decoder2_1 = myConv(dimList[3] // 4, dimList[3] // 4, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 4) // 16)
        self.decoder2_2 = myConv(dimList[3] // 4, dimList[3] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 4) // 16)
        self.decoder2_3 = myConv(dimList[3] // 8, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 8) // 16)
        self.decoder2_4 = myConv(dimList[3] // 16, 32, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)

        self.decoder2_1_up2 = upConvLayer(dimList[3]//4, dimList[3]//8, 2, norm, act, (dimList[3]//4)//16)
        self.decoder2_1_reduc2 = myConv(dimList[3]//8 + dimList[1], dimList[3]//8 - 35, kSize=1, stride=1, padding=0, bias=False,
                                        norm=norm, act=act, num_groups=(dimList[3] // 8 + dimList[1]) // 16)
        self.decoder2_1_1 = myConv(dimList[3] // 8, dimList[3] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[3] // 8) // 16)
        self.decoder2_1_2 = myConv(dimList[3] // 8, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[3] // 8) // 16)
        self.decoder2_1_3 = myConv(dimList[3] // 16, 32, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)

        self.decoder2_1_1_up3 = upConvLayer(dimList[3] // 8, dimList[3] // 16, 2, norm, act, (dimList[3] // 8) // 16)
        self.decoder2_1_1_reduc3 = myConv(dimList[3] // 16 + dimList[0], dimList[3] // 16 - 35, kSize=1, stride=1, padding=0, bias=False,
                                          norm=norm, act=act, num_groups=(dimList[3] // 16 + dimList[0]) // 16)
        self.decoder2_1_1_1 = myConv(dimList[3] // 16, dimList[3] // 16, kSize, stride=1, padding=kSize//2, bias=False,
                                     norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        self.decoder2_1_1_2 = myConv(dimList[3] // 16, 32, kSize, stride=1, padding=kSize // 2, bias=False,
                                     norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)

        self.decoder2_1_1_1_up4 = upConvLayer(dimList[3] // 16, dimList[3] // 16 - 35, 2, norm, act, (dimList[3] // 16) // 16)
        self.decoder2_1_1_1_1 = myConv(dimList[3] // 16, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                                       norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        self.decoder2_1_1_1_2 = myConv(dimList[3] // 16, dimList[3] // 32, kSize, stride=1, padding=kSize // 2, bias=False,
                                       norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        self.upscale = F.interpolate

    def forward(self, x, rgb):
        cat1, cat2, cat3, dense_feat = x[0], x[1], x[2], x[3]
        rgb_lv6, rgb_lv5, rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5]

        dense_feat = self.ASPP(dense_feat)

        lap_lv5 = torch.sigmoid(self.decoder1(dense_feat))

        lap_lv5_up = self.upscale(lap_lv5, scale_factor=2, mode='bilinear')

        dec2 = self.decoder2_up1(dense_feat)
        dec2 = self.decoder2_reduc1(torch.cat([dec2, cat3], dim=1))
        dec2_up = self.decoder2_1(torch.cat([dec2, lap_lv5_up, rgb_lv4], dim=1))
        dec2 = self.decoder2_2(dec2_up)
        dec2 = self.decoder2_3(dec2)
        lap_lv4 = torch.tanh(self.decoder2_4(dec2) + (0.1 * rgb_lv4.mean(dim=1, keepdim=True)))

        lap_lv4_up = self.upscale(lap_lv4, scale_factor=2, mode='bilinear')
        dec3 = self.decoder2_1_up2(dec2_up)
        dec3 = self.decoder2_1_reduc2(torch.cat([dec3, cat2], dim=1))
        dec3_up = self.decoder2_1_1(torch.cat([dec3, lap_lv4_up, rgb_lv3], dim=1))
        dec3 = self.decoder2_1_2(dec3_up)
        lap_lv3 = torch.tanh(self.decoder2_1_3(dec3) + (0.1 * rgb_lv3.mean(dim=1, keepdim=True)))

        lap_lv3_up = self.upscale(lap_lv3, scale_factor=2, mode='bilinear')
        dec4 = self.decoder2_1_1_up3(dec3_up)
        dec4 = self.decoder2_1_1_reduc3(torch.cat([dec4, cat1], dim=1))
        dec4_up = self.decoder2_1_1_1(torch.cat([dec4, lap_lv3_up, rgb_lv2], dim=1))
        lap_lv2 = torch.tanh(self.decoder2_1_1_2(dec4_up) + (0.1 * rgb_lv2.mean(dim=1, keepdim=True)))

        lap_lv2_up = self.upscale(lap_lv2, scale_factor=2, mode='bilinear')
        dec5 = self.decoder2_1_1_1_up4(dec4_up)
        dec5 = self.decoder2_1_1_1_1(torch.cat([dec5, lap_lv2_up, rgb_lv1], dim=1))
        dec5 = self.decoder2_1_1_1_2(dec5)
        lap_lv1 = torch.tanh(dec5 + (0.1 * rgb_lv1.mean(dim=1, keepdim=True)))


        # Laplacian restoration
        lap_lv4_img = lap_lv4 + lap_lv5_up
        lap_lv3_img = lap_lv3 + self.upscale(lap_lv4_img, scale_factor=2, mode='bilinear')
        lap_lv2_img = lap_lv2 + self.upscale(lap_lv3_img, scale_factor=2, mode='bilinear')
        final_depth = lap_lv1 + self.upscale(lap_lv2_img, scale_factor=2, mode='bilinear')
        final_depth = torch.sigmoid(final_depth)  # [12, 32, 192, 640]

        return final_depth


class ResnetEncoderDecoder(nn.Module):
    def __init__(self, num_layers=50, num_features=512, model_dim=32):
        super(ResnetEncoderDecoder, self).__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=True, num_input_images=1)
        self.decoder = Lap_decoder_lv5(dimList=[64, 256, 512, 1024])

    def forward(self, x, **kwargs):
        out_featList = self.encoder(x)

        rgb_down2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')              # 3 x H/2  x W/2
        rgb_down4 = F.interpolate(rgb_down2, scale_factor=0.5, mode='bilinear')      # 3 x H/4  x W/4
        rgb_down8 = F.interpolate(rgb_down4, scale_factor=0.5, mode='bilinear')      # 3 x H/8  x W/8
        rgb_down16 = F.interpolate(rgb_down8, scale_factor=0.5, mode='bilinear')     # 3 x H/16 x W/16
        rgb_down32 = F.interpolate(rgb_down16, scale_factor=0.5, mode='bilinear')    # 3 x H/32 x W/32
        rgb_up16 = F.interpolate(rgb_down32, rgb_down16.shape[2:], mode='bilinear')  # 3 x H/16 x W/16
        rgb_up8 = F.interpolate(rgb_down16, rgb_down8.shape[2:], mode='bilinear')    # 3 x H/8  x W/8
        rgb_up4 = F.interpolate(rgb_down8, rgb_down4.shape[2:], mode='bilinear')     # 3 x H/4  x W/4
        rgb_up2 = F.interpolate(rgb_down4, rgb_down2.shape[2:], mode='bilinear')     # 3 x H/2  x W/2
        rgb_up = F.interpolate(rgb_down2, x.shape[2:], mode='bilinear')              # 3 x H x W
        lap1 = x - rgb_up
        lap2 = rgb_down2 - rgb_up2
        lap3 = rgb_down4 - rgb_up4
        lap4 = rgb_down8 - rgb_up8
        lap5 = rgb_down16 - rgb_up16
        rgb_list = [rgb_down32, lap5, lap4, lap3, lap2, lap1]

        depth = self.decoder(out_featList, rgb_list, **kwargs)

        return depth

class Resnet50EncoderDecoder(nn.Module):
    def __init__(self, model_dim=128):
        super(Resnet50EncoderDecoder, self).__init__()
        self.encoder = ResnetEncoder(num_layers=50, pretrained=True, num_input_images=1)
        self.decoder = DecoderBN(num_features=512, num_classes=model_dim, bottleneck_features=2048)

    def forward(self, x, **kwargs):
        x = self.encoder(x)
        return self.decoder(x, **kwargs)
