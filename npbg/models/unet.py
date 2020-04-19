import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial

from npbg.models.common import get_norm_layer, Identity
from npbg.models.compose import ListModule
from npbg.models.conv import PartialConv2d


_assert_if_size_mismatch = True


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization=nn.BatchNorm2d):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                                    normalization(out_channels),
                                    nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
                                    normalization(out_channels),
                                    nn.ReLU())

    def forward(self, inputs, **kwargs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class PartialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization=nn.BatchNorm2d):
        super().__init__()

        self.conv1 = PartialConv2d(
            in_channels, out_channels, kernel_size, padding=1)

        self.conv2 = nn.Sequential(
            normalization(out_channels),
            nn.ReLU(),   
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            normalization(out_channels),
            nn.ReLU()
        )
                                       
    def forward(self, inputs, mask=None):
        outputs = self.conv1(inputs, mask)
        outputs = self.conv2(outputs)
        return outputs


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding_mode='reflect', act_fun=nn.ELU, normalization=nn.BatchNorm2d):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_f': act_fun(),
                'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_m': nn.Sigmoid(),
                'norm': normalization(out_channels)
            }
        )

    def forward(self, x, *args, **kwargs):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)

        return output


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=BasicBlock):
        super().__init__()

        self.conv = conv_block(in_channels, out_channels)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs, mask=None):
        outputs = self.down(inputs)
        outputs = self.conv(outputs, mask=mask)
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, out_channels, upsample_mode, same_num_filt=False, conv_block=BasicBlock):
        super().__init__()

        num_filt = out_channels if same_num_filt else out_channels * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(num_filt, out_channels, 4, stride=2, padding=1)
            self.conv = conv_block(out_channels * 2, out_channels, normalization=Identity)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    # Before refactoring, it was a nn.Sequential with only one module.
                                    # Need this for backward compatibility with model checkpoints.
                                    nn.Sequential(
                                        nn.Conv2d(num_filt, out_channels, 3, padding=1)
                                        )
                                    )
            self.conv = conv_block(out_channels * 2, out_channels, normalization=Identity)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)

        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            if _assert_if_size_mismatch:
                raise ValueError(f'input2 size ({inputs2.shape[2:]}) does not match upscaled inputs1 size ({in1_up.shape[2:]}')
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output= self.conv(torch.cat([in1_up, inputs2_], 1))

        return output


class UNet(nn.Module):
    r""" Rendering network with UNet architecture and multi-scale input.

    Args:
        num_input_channels: Number of channels in the input tensor or list of tensors. An integer or a list of integers for each input tensor.
        num_output_channels: Number of output channels.
        feature_scale: Division factor of number of convolutional channels. The bigger the less parameters in the model.
        more_layers: Additional down/up-sample layers.
        upsample_mode: One of 'deconv', 'bilinear' or 'nearest' for ConvTranspose, Bilinear or Nearest upsampling.
        norm_layer: [unused] One of 'bn', 'in' or 'none' for BatchNorm, InstanceNorm or no normalization. Default: 'bn'.
        last_act: Last layer activation. One of 'sigmoid', 'tanh' or None.
        conv_block: Type of convolutional block, like Convolution-Normalization-Activation. One of 'basic', 'partial' or 'gated'.
    """
    def __init__(
        self,
        num_input_channels=3, 
        num_output_channels=3,
        feature_scale=4,
        more_layers=0,
        upsample_mode='bilinear',
        norm_layer='bn',
        last_act='sigmoid',
        conv_block='partial'
    ):
        super().__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers

        if isinstance(num_input_channels, int):
            num_input_channels = [num_input_channels]

        if len(num_input_channels) < 5:
            num_input_channels += [0] * (5 - len(num_input_channels))
        
        self.num_input_channels = num_input_channels[:5]

        if conv_block == 'basic':
            self.conv_block = BasicBlock
        elif conv_block == 'partial':
            self.conv_block = PartialBlock
        elif conv_block == 'gated':
            self.conv_block = GatedBlock
        else:
            raise ValueError('bad conv block {}'.format(conv_block))

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        # norm_layer = get_norm_layer(norm_layer)

        self.start = self.conv_block(self.num_input_channels[0], filters[0])

        self.down1 = DownsampleBlock(filters[0], filters[1] - self.num_input_channels[1], conv_block=self.conv_block)
        self.down2 = DownsampleBlock(filters[1], filters[2] - self.num_input_channels[2], conv_block=self.conv_block)
        self.down3 = DownsampleBlock(filters[2], filters[3] - self.num_input_channels[3], conv_block=self.conv_block)
        self.down4 = DownsampleBlock(filters[3], filters[4] - self.num_input_channels[4], conv_block=self.conv_block)

        if self.more_layers > 0:
            self.more_downs = [
                DownsampleBlock(filters[4], filters[4], conv_block=self.conv_block) for i in range(self.more_layers)]
            self.more_ups = [UpsampleBlock(filters[4], upsample_mode, same_num_filt =True, conv_block=self.conv_block) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        self.up4 = UpsampleBlock(filters[3], upsample_mode, conv_block=self.conv_block)
        self.up3 = UpsampleBlock(filters[2], upsample_mode, conv_block=self.conv_block)
        self.up2 = UpsampleBlock(filters[1], upsample_mode, conv_block=self.conv_block)
        self.up1 = UpsampleBlock(filters[0], upsample_mode, conv_block=self.conv_block)

        # Before refactoring, it was a nn.Sequential with only one module.
        # Need this for backward compatibility with model checkpoints.
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], num_output_channels, 1)
        )

        if last_act == 'sigmoid':
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        elif last_act == 'tanh':
            self.final = nn.Sequential(self.final, nn.Tanh())

    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)

        if isinstance(self.conv_block, PartialBlock):
            eps = 1e-9
            masks = [(x.sum(1) > eps).float() for x in inputs]
        else:
            masks = [None] * len(inputs)

        n_input = len(inputs)
        n_declared = np.count_nonzero(self.num_input_channels)
        assert n_input == n_declared, f'got {n_input} input scales but declared {n_declared}'

        in64 = self.start(inputs[0], mask=masks[0])
        
        mask = masks[1] if self.num_input_channels[1] else None
        down1 = self.down1(in64, mask)
        
        
        if self.num_input_channels[1]:
            down1 = torch.cat([down1, inputs[1]], 1)
        
        mask = masks[2] if self.num_input_channels[2] else None
        down2 = self.down2(down1, mask)
        
        if self.num_input_channels[2]:
            down2 = torch.cat([down2, inputs[2]], 1)
        
        mask = masks[3] if self.num_input_channels[3] else None
        down3 = self.down3(down2, mask)
        
        if self.num_input_channels[3]:
            down3 = torch.cat([down3, inputs[3]], 1)
        
        mask = masks[4] if self.num_input_channels[4] else None
        down4 = self.down4(down3, mask)
        if self.num_input_channels[4]:
            down4 = torch.cat([down4, inputs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_= l(up_, prevs[self.more - idx - 2])
        else:
            up_= down4

        up4 = self.up4(up_, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)
        
        return self.final(up1)
