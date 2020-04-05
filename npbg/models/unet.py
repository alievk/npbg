import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial

from npbg.models.common import norm
from npbg.models.compose import ListModule
from npbg.models.conv import conv, PartialConv2d, GatedConv2d


conv_fn = None
_assert_if_size_mismatch = True


class UNet(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=3,
                       feature_scale=4, more_layers=0,
                       upsample_mode='bilinear', pad='zero', norm_layer='bn',
                       last_act='sigmoid', need_bias=True,
                       conv_class=nn.Conv2d, conv_block='partial'):
        super(UNet, self).__init__()

        if isinstance(num_input_channels, int):
            num_input_channels = [num_input_channels]

        if len(num_input_channels) < 5:
            num_input_channels += [0] * (5 - len(num_input_channels))
        
        num_input_channels = num_input_channels[:5]

        self.feature_scale = feature_scale
        self.more_layers = more_layers

        if conv_block == 'basic':
            self.conv_block = UNetConv
        elif conv_block == 'partial':
            self.conv_block = UNetConvPartial
        elif conv_block == 'gated':
            self.conv_block = UNetConvGated
        else:
            raise ValueError('bad conv block {}'.format(conv_block))

        global conv_fn
        conv_fn = partial(conv, bias=need_bias, pad=pad, conv_class=conv_class)


        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        self.start = self.conv_block(num_input_channels[0], filters[0], norm_layer, need_bias, pad)

        self.down1 = UNetDown(filters[0], filters[1] - num_input_channels[1], norm_layer, need_bias, pad, conv_block=self.conv_block)
        self.down2 = UNetDown(filters[1], filters[2] - num_input_channels[2], norm_layer, need_bias, pad, conv_block=self.conv_block)
        self.down3 = UNetDown(filters[2], filters[3] - num_input_channels[3], norm_layer, need_bias, pad, conv_block=self.conv_block)
        self.down4 = UNetDown(filters[3], filters[4] - num_input_channels[4], norm_layer, need_bias, pad, conv_block=self.conv_block)

        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                UNetDown(filters[4], filters[4], norm_layer, need_bias, pad, conv_block=self.conv_block) for i in range(self.more_layers)]
            self.more_ups = [UNetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt =True, conv_block=self.conv_block) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        self.up4 = UNetUp(filters[3], upsample_mode, need_bias, pad, conv_block=self.conv_block)
        self.up3 = UNetUp(filters[2], upsample_mode, need_bias, pad, conv_block=self.conv_block)
        self.up2 = UNetUp(filters[1], upsample_mode, need_bias, pad, conv_block=self.conv_block)
        self.up1 = UNetUp(filters[0], upsample_mode, need_bias, pad, conv_block=self.conv_block)

        self.final = conv_fn(filters[0], num_output_channels, 1)

        if last_act == 'sigmoid':
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        elif last_act == 'tanh':
            self.final = nn.Sequential(self.final, nn.Tanh())

        self.num_input_channels = num_input_channels

    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)

        if isinstance(self.conv_block, UNetConvPartial):
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

        up4= self.up4(up_, down3)
        up3= self.up3(up4, down2)
        up2= self.up2(up3, down1)
        up1= self.up1(up2, in64)
        
        return self.final(up1)



class UNetConv(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super().__init__()

        # print(pad)
        if norm_layer is not None:
            self.conv1= nn.Sequential(conv_fn(in_size, out_size, 3),
                                       norm(out_size, norm_layer),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv_fn(out_size, out_size, 3),
                                       norm(out_size, norm_layer),
                                       nn.ReLU(),)
        else:
            self.conv1= nn.Sequential(conv_fn(in_size, out_size, 3),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv_fn(out_size, out_size, 3),
                                       nn.ReLU(),)
    def forward(self, inputs, **kwargs):
        outputs= self.conv1(inputs)
        outputs= self.conv2(outputs)
        return outputs


class UNetConvPartial(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(UNetConvPartial, self).__init__()

        # print(pad)
        if norm_layer is not None:
            self.conv1=PartialConv2d(in_size, out_size, 3, padding=1)
            self.conv2= nn.Sequential(
                                        norm(out_size, norm_layer),
                                        nn.ReLU(),
                                        conv_fn(out_size, out_size, 3),
                                        norm(out_size, norm_layer),
                                        nn.ReLU(),)
        else:
            self.conv1= PartialConv2d(in_size, out_size, 3, padding=1)
            self.conv2= nn.Sequential(nn.ReLU(),
                                        conv_fn(out_size, out_size, 3),
                                       nn.ReLU(),)
    def forward(self, inputs, mask=None):
#         if mask is not None:
#             print(mask.shape)
        outputs= self.conv1(inputs, mask)
        outputs= self.conv2(outputs)
        return outputs


class UNetConvGated(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super().__init__()
        if norm_layer == 'bn':
            norm_cls = nn.BatchNorm2d
        elif norm_layer in ('none', None):
            norm_cls = None
        else:
            raise ValueError(f'{norm_layer} not supported')
        self.block = GatedConv2d(in_size, out_size, normalization=norm_cls)

    def forward(self, x, **kwargs):
        return self.block(x)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, conv_block=UNetConv):
        super(UNetDown, self).__init__()
        self.conv= conv_block(in_size, out_size, norm_layer, need_bias, pad)
        # self.down= nn.MaxPool2d(2, 2)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs, mask=None):
        outputs= self.down(inputs)
        outputs= self.conv(outputs, mask=mask)
        return outputs


class UNetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False, conv_block=UNetConv):
        super(UNetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= conv_block(out_size * 2, out_size, None, need_bias, pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    conv_fn(num_filt, out_size, 3, bias=need_bias))
            self.conv= conv_block(out_size * 2, out_size, None, need_bias, pad)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up= self.up(inputs1)

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
