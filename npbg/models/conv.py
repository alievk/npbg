import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride', conv_class=nn.Conv2d):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection' and to_pad != 0:
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = conv_class(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    # print (layers22)
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    # print (layers)
    return nn.Sequential(*layers)


class GatedConv2d(nn.Module):
    def __init__(self, in_features, out_features, filter_size=3, stride=1, dilation=1,
                 padding_mode='reflect', normalization=nn.BatchNorm2d, act_fun=nn.ELU):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = filter_size
        self.stride = stride
        self.dilation = dilation

        self.conv_f = nn.Conv2d(in_features, out_features, filter_size, stride=stride, dilation=dilation)
        if normalization is not None:
            self.norm = normalization(out_features)
        else:
            self.norm = None
        self.act_f = act_fun()

        self.conv_m = nn.Conv2d(in_features, out_features, filter_size, stride=stride, dilation=dilation)
        self.act_m = nn.Sigmoid()

    def forward(self, x):
        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)
        n_pad_by_sides = (n_pad_pxl, n_pad_pxl, n_pad_pxl, n_pad_pxl)
        x_padded = F.pad(x, n_pad_by_sides, mode=self.pad_mode)

        features = self.act_f(self.conv_f(x_padded))
        mask = self.act_m(self.conv_m(x_padded))
        output = features * mask
        if self.norm is not None:
            output = self.norm(output)

        return output


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):

        if mask_in is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output