import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import conv, norm, ListModule
import numpy as np
from functools import partial

from models.partial_conv import PartialConv2d

def get_native_transform():
    return Identity

def get_sins(world_batch, sin_weights):
    # world_batch in [0, 1] or similar

    lin_combs = torch.sum(world_batch[:, None, :, :, :] * sin_weights[None, :, :, None, None], 2)

    sins = torch.sin(lin_combs) * 0.5 + 0.5

    return sins

conv_fn = None

def get_conv_block(conv_block):
    block = None
    if conv_block == 'basic':
        block = unetConv2
    elif conv_block == 'partial':
        block = unetConv2_part
    elif conv_block == 'gated':
        block = gatedConvWrapper
    return block

class UNet(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3,
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode='deconv', pad='zero', norm_layer='in',
                       last_act='sigmoid', need_bias=True,
                       conv_class_fn=nn.Conv2d, modality=2,
                       conv_block='partial', conv_block_up='gated', aux_inputs=[0, 0, 0, 0]):
        super().__init__()
        print('need_bias:', need_bias)

        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x
        self.modality = modality
        self.conv_block_type = conv_block

        self.conv_block = get_conv_block(conv_block)
        self.conv_block_up = get_conv_block(conv_block_up)
            
            

        global conv_fn
        conv_fn = partial(conv,  bias=need_bias, pad=pad, conv_class=conv_class_fn)


        filters = [64, 128, 256, 512, 1024]

        filters = [x // self.feature_scale for x in filters]

        num_concat_ch = self.modality * 3

        self.start = self.conv_block(num_input_channels, filters[0] if not concat_x else filters[0] - num_concat_ch, norm_layer, need_bias, pad)
        
        self.down1 = unetDown(filters[0], filters[1] if not aux_inputs[0] else filters[1] - num_input_channels, norm_layer, need_bias, pad, conv_block=self.conv_block)
        self.down2 = unetDown(filters[1], filters[2] if not aux_inputs[1] else filters[2] - num_input_channels, norm_layer, need_bias, pad, conv_block=self.conv_block)
        self.down3 = unetDown(filters[2], filters[3] if not aux_inputs[2] else filters[3] - num_input_channels, norm_layer, need_bias, pad, conv_block=self.conv_block)
        self.down4 = unetDown(filters[3], filters[4] if not aux_inputs[3] else filters[4] - num_input_channels, norm_layer, need_bias, pad, conv_block=self.conv_block)

        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels , norm_layer, need_bias, pad, conv_block=self.conv_block) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt =True, conv_block=self.conv_block) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad, conv_block=self.conv_block)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad, conv_block=self.conv_block)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad, conv_block=self.conv_block)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad, conv_block=self.conv_block)
        
#         print('filters[0], num_output_channels', filters[0], num_output_channels)
        self.final = conv_fn(filters[0], num_output_channels, 1)

        if last_act == 'sigmoid':
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        elif last_act == 'tanh':
            self.final = nn.Sequential(self.final, nn.Tanh())

    def forward(self, inputs, masks=None):
        if self.concat_x:
            # Downsample
            downs = []
            for i in range(4 + self.more_layers + 1):
                start, end = self.modality * i * 3, self.modality * (i + 1) * 3
                x = inputs[:, start:end]
                if i:
                    ks = 2 ** i
                    x = nn.AvgPool2d(ks, ks)(x)
                    downs.append(x)
                else:
                    downs.append(x)

        in64 = self.start(inputs['x/1'], masks=masks['x/1'])
#         print(in64.shape)
#         if 'x/2' in inputs:
#             in64 = torch.cat([in64, inputs['x/2']], 1)
        masks1 = None
        if 'x/2' in inputs:
            masks1 = masks['x/2']
#         else:
#             masks1 = F.interpolate(masks['x/1'], scale_factor=0.5)
#         print('masks1', masks1.shape)
        down1 = self.down1(in64, masks=masks1)
#         print(down1.shape)

        if 'x/2' in inputs:
#             print(inputs['x/2'].shape)
            down1 = torch.cat([down1, inputs['x/2']], 1)
        masks2 = None   
        if 'x/4' in inputs:
            masks2 = masks['x/4']
#         else:
#             masks2 =  F.interpolate(masks1, scale_factor=0.5)
#             print(down1.shape)
#         print(down1.shape, masks1.shape)
        down2 = self.down2(down1, masks=masks2)
#         print(down2.shape)
        
        if 'x/4' in inputs:
#             print(inputs['x/4'].shape)
            down2 = torch.cat([down2, inputs['x/4']], 1)
#             masks2 = masks['x/4']
        
#             print(down2.shape)
#         print('masks2', masks2.shape)
        masks3 = None
        if 'x/8' in inputs:
            masks3 = masks['x/8']
#         else:
#             masks3 =  F.interpolate(masks2, scale_factor=0.5)
            
        down3 = self.down3(down2, masks=masks3)
#         print(down3.shape)
        
        if 'x/8' in inputs:
            down3 = torch.cat([down3, inputs['x/8']], 1)
            
#         print('masks3', masks3.shape)
        masks4 = None
        if 'x/16' in inputs:
            masks4 = masks['x/16']
#         else:
#             masks4 =  F.interpolate(masks3, scale_factor=0.5)
        down4 = self.down4(down3, masks=masks4)
#         print(down4.shape)
       
        if 'x/16' in inputs:
            down4 = torch.cat([down4, inputs['x/16']], 1)
#         print('masks4', masks4.shape)
        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                # print(prevs[-1].size())
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out,  downs[kk + 5]], 1)

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
        
        #up3= self.up3(down3, down2)
        #up2= self.up2(up3, down1)
        #up1= self.up1(up2, in64)
        
        #up2= self.up2(down2, down1)
        #up1= self.up1(up2, in64)
        
        #up1= self.up1(down1, in64)
        
        #up1=in64

        return self.final(up1)



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()

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
    def forward(self, inputs):
        outputs= self.conv1(inputs)
        outputs= self.conv2(outputs)
        return outputs


class unetConv2_part(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2_part, self).__init__()

        # print(pad)
        if norm_layer is not None:
            self.conv1=PartialConv2d(in_size, out_size, 3, padding=1, bias=need_bias)
            self.conv2= nn.Sequential(
                                        norm(out_size, norm_layer),
                                        nn.ReLU(),
                                        conv_fn(out_size, out_size, 3),
                                        norm(out_size, norm_layer),
                                        nn.ReLU(),)
        else:
            self.conv1= PartialConv2d(in_size, out_size, 3, padding=1, bias=need_bias)
            self.conv2= nn.Sequential(nn.ReLU(),
                                        conv_fn(out_size, out_size, 3),
                                       nn.ReLU(),)
    def forward(self, inputs, masks=None):
#         print(inputs.shape, masks.shape)
        outputs= self.conv1(inputs, masks)
        outputs= self.conv2(outputs)
        return outputs


class _gated_conv_block(nn.Module):
    def __init__(self, in_features, out_features, filter_size=3, stride=1, dilation=1,
                 padding_mode='reflect', sn=False, normalization=nn.BatchNorm2d, act_fun=nn.ELU):
        super(_gated_conv_block, self).__init__()
        self.pad_mode = padding_mode
        self.filter_size = filter_size
        self.stride = stride
        self.dilation = dilation

        self.conv_f = nn.Conv2d(in_features, out_features, filter_size, stride=stride, dilation=dilation)
        if sn:
            self.conv_f = SpectralNorm(self.conv_f)
        if normalization is not None:
            self.norm = normalization(out_features)
        else:
            self.norm = None
        self.act_f = act_fun()

        self.conv_m = nn.Conv2d(in_features, out_features, filter_size, stride=stride, dilation=dilation)
        if sn:
            self.conv_m = SpectralNorm(self.conv_m)
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


class gatedConvWrapper(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(gatedConvWrapper, self).__init__()
        if norm_layer == 'bn':
            norm_cls = nn.BatchNorm2d
        elif norm_layer == None:
            norm_cls = None
        else:
            raise ValueError('not supported')
        self.block = _gated_conv_block(in_size, out_size, normalization=norm_cls)

    def forward(self, x):
        return self.block(x)


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, conv_block=unetConv2):
        super(unetDown, self).__init__()
        self.conv= conv_block(in_size, out_size, norm_layer, need_bias, pad)
        # self.down= nn.MaxPool2d(2, 2)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs, masks=None):
        outputs = self.down(inputs)
        outputs = self.conv(outputs, masks)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False, conv_block=unetConv2):
        super(unetUp, self).__init__()

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
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output= self.conv(torch.cat([in1_up, inputs2_], 1))

        return output
