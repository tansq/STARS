import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

sys.path.append('..')
from tools.utils import *


OPS = {
    'MBI_k3_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
    'MBI_k3_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
    'MBI_k5_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
    'MBI_k5_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
    'MBI_k3_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 3, s, affine=aff, act_func=act),
    'MBI_k3_e6_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 3, s, affine=aff, act_func=act),
    'MBI_k5_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 5, s, affine=aff, act_func=act),
    'MBI_k5_e6_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 5, s, affine=aff, act_func=act),
    'BasicRes_k3': lambda ic, mc, oc, s, aff, act: BasicResBlock(ic, 0, oc, 3, s, affine=aff, act_func=act),
    'BasicRes_k5': lambda ic, mc, oc, s, aff, act: BasicResBlock(ic, 0, oc, 5, s, affine=aff, act_func=act),
    'UCT3Res_k3': lambda ic, mc, oc, s, aff, act: UCT3ResBlock(ic, 0, oc, 3, s, affine=aff, act_func=act),
    'SRRes_k3': lambda ic, mc, oc, s, aff, act: SRBlock(ic, 0, oc, 3, s, affine=aff, act_func=act),
    # 'skip'      : lambda ic, mc, oc, s, aff, act: IdentityLayer(ic, oc),
}

def set_layer_from_config(layer_config):
    if layer_config is None:
        return None
    
    name2layer = {
        ConvLayer.__name__: ConvLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MBInvertedResBlock.__name__: MBInvertedResBlock,
        BasicResBlock.__name__: BasicResBlock,
        SRBlock.__name__: SRBlock,
        UCT3ResBlock.__name__: UCT3ResBlock,
        SepConvLayer.__name__: SepConvLayer,
    }
    
    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(x.sigmoid())
        else:
            return x * x.sigmoid()


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(F.relu6(x + 3., inplace=True) / 6.)
        else:
            return x * F.relu6(x + 3.) /6.


def get_act_fn(act_func):

    if act_func == 'relu':
        act = nn.ReLU
    elif act_func == 'relu6':
        act = nn.ReLU6
    elif act_func == 'swish':
        act = Swish
    elif act_func == 'h-swish':
        act = HardSwish
    else:
        act = None
    return act


class BasicUnit(nn.Module):

    def forward(self, x):
        raise NotImplementedError
    
    @property
    def name(self):
        raise NotImplementedError
    
    @property
    def unit_str(self):
        raise NotImplementedError
    
    @property
    def config(self):
        raise NotImplementedError
    
    @staticmethod
    def build_from_config(config):
        raise NotImplementedError
    
    def get_flops(self, x):
        raise NotImplementedError
    
    def get_latency(self, x):
        raise NotImplementedError


class BasicLayer(BasicUnit):

    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        affine = True,
        act_func='relu6',
        ops_order='weight_bn_act'):
        super(BasicLayer, self).__init__()
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.affine = affine
        self.act_func = act_func
        self.ops_order = ops_order
        
        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm2d(in_channels, affine=affine, track_running_stats=affine)
            else:
                self.bn = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
        else:
            self.bn = None
        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.act = nn.ReLU(inplace=False)
            else:
                self.act = nn.ReLU(inplace=True)
        elif act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.act = nn.ReLU6(inplace=False)
            else:
                self.act = nn.ReLU6(inplace=True)
        elif act_func == 'swish':
            if self.ops_list[0] == 'act':
                self.act = Swish(inplace=False)
            else:
                self.act = Swish(inplace=True)
        elif act_func == 'h-swish':
            if self.ops_list[0] == 'act':
                self.act = HardSwish(inplace=False)
            else:
                self.act = HardSwish(inplace=True)
        else:
            self.act = None

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                x = self.weight_call(x)
            elif op == 'bn':
                if self.bn is not None:
                    x = self.bn(x)
            elif op == 'act':
                if self.act is not None:
                    x = self.act(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x

    @property
    def name(self):
        raise NotImplementedError

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'affine': self.affine,
            'act_func': self.act_func,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError
    
    def get_flops(self):
        raise NotImplementedError
    
    def get_latency(self):
        raise NotImplementedError


class ConvLayer(BasicLayer):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        has_shuffle=False,
        bias=False,
        use_bn=True,
        affine=True,
        act_func='relu6',
        ops_order='weight_bn_act'):
        super(ConvLayer, self).__init__(
    	    in_channels,
    	    out_channels,
    	    use_bn,
    	    affine,
    	    act_func,
    	    ops_order)
    
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.has_shuffle = has_shuffle
        self.bias = bias
        
        padding = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            groups=self.groups,
            bias=self.bias)

    def weight_call(self, x):
        x = self.conv(x)
        if self.has_shuffle and self.groups > 1:
            x = channel_shuffle(x, self.groups)
        return x

    @property
    def name(self):
        return ConvLayer.__name__
    
    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            return '%dx%d_GroupConv_G%d' % (kernel_size[0], kernel_size[1], self.groups)
    
    @property
    def config(self):
        config = {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'groups': self.groups,
            'has_shuffle': self.has_shuffle,
            'bias': self.bias,
        }
        config.update(super(ConvLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)
    
    def get_flops(self):
        raise NotImplementedError
    
    def get_latency(self):
        raise NotImplementedError


class IdentityLayer(BasicLayer):

    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=False,
        affine=False,
        act_func=None,
        ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(
            in_channels,
            out_channels,
            use_bn,
            affine,
            act_func,
            ops_order)

    def weight_call(self, x):
        return x
    
    @property
    def name(self):
        return IdentityLayer.__name__
    
    @property
    def unit_str(self):
        return 'Identity'
    
    @property
    def config(self):
        config = {
            'name': IdentityLayer.__name__,
        }
        config.update(super(IdentityLayer, self).config)
        return config
    
    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)
    
    def get_flops(self):
        raise NotImplementedError
    
    def get_latency(self):
        raise NotImplementedError


class LinearLayer(BasicUnit):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        use_bn=False,
        affine=False,
        act_func=None,
        ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.use_bn = use_bn
        self.affine = affine
        self.act_func = act_func
        self.ops_order = ops_order
        
        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm1d(in_features, affine=affine, track_running_stats=affine)
            else:
                self.bn = nn.BatchNorm1d(out_features, affine=affine, track_running_stats=affine)
        else:
            self.bn = None
        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.act = nn.ReLU(inplace=False)
            else:
                self.act = nn.ReLU(inplace=True)
        elif act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.act = nn.ReLU6(inplace=False)
            else:
                self.act = nn.ReLU6(inplace=True)
        elif act_func == 'tanh':
            self.act = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None
        # linear
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)
    
    @property
    def ops_list(self):
        return self.ops_order.split('_')
    
    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)
    
    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                x = self.linear(x)
            elif op == 'bn':
                if self.bn is not None:
                    x = self.bn(x)
            elif op == 'act':
                if self.act is not None:
                    x = self.act(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x
    
    @property
    def name(self):
        return LinearLayer.__name__
    
    @property
    def unit_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)
    
    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'affine': self.affine,
            'act_func': self.act_func,
            'ops_order': self.ops_order,
        }
    
    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)
    
    def get_flops(self):
        raise NotImplementedError
    
    def get_latency(self):
        raise NotImplementedError


# use for color steg stem layer
class SepConvLayer(BasicUnit):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            has_shuffle=False,
            bias=False,
            use_bn=True,
            affine=True,
            act_func='relu'):
        super(SepConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.has_shuffle = has_shuffle
        self.bias = bias
        self.use_bn = use_bn
        self.affine = affine
        self.act_func = act_func

        assert out_channels % in_channels == 0, "inc={}, ouc={}".format(in_channels, out_channels)
        assert in_channels == 3

        padding = get_same_padding(self.kernel_size)
        if use_bn:
            bn = nn.BatchNorm2d
        else:
            bn = nn.Identity

        act = get_act_fn(act_func)
        if act is None:
            act = nn.Identity

        self.conv = nn.Conv2d(1, out_channels // in_channels, kernel_size, stride,
                       padding, bias=bias)
        # if bn is not None:
        self.bn = bn(out_channels // in_channels, affine=affine, track_running_stats=affine)
        # if act is not None:
        self.act = act(inplace=True)

    def forward(self, x):

        output_c1 = x[:, 0, :, :]
        output_c2 = x[:, 1, :, :]
        output_c3 = x[:, 2, :, :]
        out_c1 = output_c1.unsqueeze(1)
        out_c2 = output_c2.unsqueeze(1)
        out_c3 = output_c3.unsqueeze(1)
        c1 = self.act(self.bn(self.conv(out_c1)))
        c2 = self.act(self.bn(self.conv(out_c2)))
        c3 = self.act(self.bn(self.conv(out_c3)))
        out = torch.cat([c1, c2, c3], dim=1)  # 3*30=90

        return out

    @property
    def name(self):
        return SepConvLayer.__name__

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_SepConvLayer' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': SepConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'groups': self.groups,
            'has_shuffle': self.has_shuffle,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'affine': self.affine,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return SepConvLayer(**config)

    def get_flops(self):
        raise NotImplementedError

    def get_latency(self):
        raise NotImplementedError


class BasicResBlock(BasicUnit):

    def __init__(
        self,
        in_channels,
        # mid_channels,
        se_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        has_shuffle=False,
        bias=False,
        use_bn=True,
        affine=True,
        act_func='relu'):
        super(BasicResBlock, self).__init__()
        
        self.in_channels = in_channels
        self.se_channels = se_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.has_shuffle = has_shuffle
        self.bias = bias
        self.use_bn = use_bn
        self.affine = affine
        self.act_func = act_func
        
        padding = get_same_padding(self.kernel_size)
        if use_bn:
            bn = nn.BatchNorm2d
        else:
            bn = None

        act = get_act_fn(act_func)
        
        conv1 = OrderedDict([
                ('conv', 
                 nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, bias=bias)),
            ])
        if bn is not None:
            conv1['bn'] = bn(out_channels, affine=affine, track_running_stats=affine)
        if act is not None:
            conv1['act'] = act(inplace=True)
        self.conv1 = nn.Sequential(conv1)
        
        conv2 = OrderedDict([
            ('conv', 
             nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                  padding, bias=bias)),
            ])
        if bn is not None:
            conv2['bn'] = bn(out_channels, affine=affine, track_running_stats=affine)
        self.conv2 = nn.Sequential(conv2)
        # self.act = act(inplace=True)
        
        # se model
        if se_channels > 0:
            squeeze_excite = OrderedDict([
                    ('conv_reduce', nn.Conv2d(out_channels, se_channels, 1, 1, 0, groups=groups, bias=True)),
                ])
            if act is not None:
                squeeze_excite['act'] = act(inplace=True)
            squeeze_excite['conv_expand'] = nn.Conv2d(se_channels, out_channels, 1, 1, 0, groups=groups, bias=True)
            self.squeeze_excite = nn.Sequential(squeeze_excite)
        else:
            self.squeeze_excite = None
        
        # # residual flag
        # self.has_residual = (in_channels == out_channels) and (stride == 1)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)),
                ])
            if bn is not None:
                downsample['bn'] = bn(out_channels, affine=affine, track_running_stats=affine)
            self.downsample = nn.Sequential(downsample)
    
    def forward(self, x):
        res = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        if self.squeeze_excite is not None:
            x_se = F.adaptive_avg_pool2d(x, 1)
            x = x * torch.sigmoid(self.squeeze_excite(x_se))
        
        if self.has_shuffle and self.groups > 1:
            x = channel_shuffle(x, self.groups)

        if self.downsample is not None:
            res = self.downsample(res)
        x += res
        # x = self.act(x)
        
        return x
    
    @property
    def name(self):
        return BasicResBlock.__name__
    
    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            return '%dx%d_BasicResBlock' % (kernel_size[0], kernel_size[1])
        else:
            return '%dx%d_BasicResBlock_G%d' % (kernel_size[0], kernel_size[1],
    													self.groups)
    
    @property
    def config(self):
        return {
            'name': BasicResBlock.__name__,
            'in_channels': self.in_channels,
            # 'mid_channels': self.mid_channels,
            'se_channels': self.se_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'groups': self.groups,
            'has_shuffle': self.has_shuffle,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'affine': self.affine,
            'act_func': self.act_func,
        }
    
    @staticmethod
    def build_from_config(config):
        return BasicResBlock(**config)
    
    
    def get_flops(self):
        raise NotImplementedError
    
    def get_latency(self):
        raise NotImplementedError


class SRBlock(BasicUnit):
    # from srnet
    def __init__(
            self,
            in_channels,
            # mid_channels,
            se_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            has_shuffle=False,
            bias=False,
            use_bn=True,
            affine=True,
            act_func='relu'):
        super(SRBlock, self).__init__()

        self.in_channels = in_channels
        self.se_channels = se_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.has_shuffle = has_shuffle
        self.bias = bias
        self.use_bn = use_bn
        self.affine = affine
        self.act_func = act_func

        padding = get_same_padding(self.kernel_size)
        if use_bn:
            bn = nn.BatchNorm2d
        else:
            bn = None
        # else:
        #     bn = nn.Identity
        act = get_act_fn(act_func)

        conv1 = OrderedDict([
            ('conv',
             nn.Conv2d(in_channels, out_channels, kernel_size, 1,
                       padding, bias=bias)),
        ])
        if bn is not None:
            conv1['bn'] = bn(out_channels, affine=affine, track_running_stats=affine)
        if act is not None:
            conv1['act'] = act(inplace=True)
        self.conv1 = nn.Sequential(conv1)

        conv2 = OrderedDict([
            ('conv',
             nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                       padding, bias=bias)),
        ])
        if bn is not None:
            conv2['bn'] = bn(out_channels, affine=affine, track_running_stats=affine)
        if stride > 1:
            conv2['pool'] = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(conv2)
        # self.act = act(inplace=True)

        # # residual flag
        # self.has_residual = (in_channels == out_channels) and (stride == 1)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)),
            ])
            if bn is not None:
                downsample['bn'] = bn(out_channels, affine=affine, track_running_stats=affine)
            self.downsample = nn.Sequential(downsample)

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            res = self.downsample(res)
        x += res

        return x

    @property
    def name(self):
        return SRBlock.__name__

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        # if self.groups == 1:
        #     return '%dx%d_SRBlock' % (kernel_size[0], kernel_size[1])
        # else:
        #     return '%dx%d_SRBlock_G%d' % (kernel_size[0], kernel_size[1], self.groups)
        return '%dx%d_SRBlock' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': SRBlock.__name__,
            'in_channels': self.in_channels,
            # 'mid_channels': self.mid_channels,
            'se_channels': self.se_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'groups': self.groups,
            'has_shuffle': self.has_shuffle,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'affine': self.affine,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return SRBlock(**config)

    def get_flops(self):
        raise NotImplementedError

    def get_latency(self):
        raise NotImplementedError


class UCT3ResBlock(BasicUnit):

    def __init__(
            self,
            in_channels,
            # mid_channels,
            se_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            has_shuffle=False,
            bias=False,
            use_bn=True,
            affine=True,
            act_func='relu6'):
        super(UCT3ResBlock, self).__init__()

        self.in_channels = in_channels
        # self.mid_channels = mid_channels
        self.se_channels = se_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.has_shuffle = has_shuffle
        self.bias = bias
        self.use_bn = use_bn
        self.affine = affine
        self.act_func = act_func

        act = get_act_fn(act_func)

        # inverted bottleneck
        inverted_bottleneck = OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=groups, bias=bias)),
        ])
        if use_bn:
            inverted_bottleneck['bn'] = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
        if act is not None:
            inverted_bottleneck['act'] = act(inplace=True)
        self.inverted_bottleneck = nn.Sequential(inverted_bottleneck)

        # depthwise convolution 深度卷积，不会改变通道数，group=通道数
        padding = get_same_padding(self.kernel_size)
        depth_conv = OrderedDict([
            ('conv',
             nn.Conv2d(
                 out_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=out_channels,
                 bias=bias)),
        ])
        if use_bn:
            depth_conv['bn'] = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
        if act is not None:
            depth_conv['act'] = act(inplace=True)
        self.depth_conv = nn.Sequential(depth_conv)

        self.squeeze_excite = None
        self.se_channels = 0

        # pointwise linear
        point_linear = OrderedDict([
            ('conv', nn.Conv2d(out_channels, out_channels, 1, 1, 0, groups=groups, bias=bias)),
        ])
        if use_bn:
            point_linear['bn'] = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
        self.point_linear = nn.Sequential(point_linear)

        # residual flag
        # self.has_residual = (in_channels == out_channels) and (stride == 1)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)),
            ])
            if use_bn:
                downsample['bn'] = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
            self.downsample = nn.Sequential(downsample)

    def forward(self, x):
        res = x

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
            if self.has_shuffle and self.groups > 1:
                x = channel_shuffle(x, self.groups)

        x = self.depth_conv(x)
        if self.squeeze_excite is not None:
            x_se = F.adaptive_avg_pool2d(x, 1)
            x = x * torch.sigmoid(self.squeeze_excite(x_se))

        x = self.point_linear(x)
        if self.has_shuffle and self.groups > 1:
            x = channel_shuffle(x, self.groups)

        if self.downsample is not None:
            res = self.downsample(res)
        x += res

        return x

    @property
    def name(self):
        return UCT3ResBlock.__name__

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_UCT3ResBlock' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': UCT3ResBlock.__name__,
            'in_channels': self.in_channels,
            # 'mid_channels': self.mid_channels,
            'se_channels': self.se_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'groups': self.groups,
            'has_shuffle': self.has_shuffle,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'affine': self.affine,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return UCT3ResBlock(**config)

    def get_flops(self):
        raise NotImplementedError

    def get_latency(self):
        raise NotImplementedError


# a DepthwiseSeperableConv when mid_channels == in_channels
class MBInvertedResBlock(BasicUnit):

    def __init__(
        self,
        in_channels,
        mid_channels,
        se_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        has_shuffle=False,
        bias=False,
        use_bn=True,
        affine=True,
        act_func='relu6'):
        super(MBInvertedResBlock, self).__init__()
    
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.se_channels = se_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.has_shuffle = has_shuffle
        self.bias = bias
        self.use_bn = use_bn
        self.affine = affine
        self.act_func = act_func
        self.drop_connect_rate = 0.0

        act = get_act_fn(act_func)
        
        # inverted bottleneck
        if mid_channels > in_channels:
            inverted_bottleneck = OrderedDict([
                    ('conv', nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=bias)),
                ])
            if use_bn:
                inverted_bottleneck['bn'] = nn.BatchNorm2d(mid_channels, affine=affine, track_running_stats=affine)
            if act is not None:
                inverted_bottleneck['act'] = act(inplace=True)
            self.inverted_bottleneck = nn.Sequential(inverted_bottleneck)
        else:
            self.inverted_bottleneck = None
            self.mid_channels = in_channels
            mid_channels = in_channels
        
        # depthwise convolution 深度卷积，不会改变通道数，group=通道数
        padding = get_same_padding(self.kernel_size)
        depth_conv = OrderedDict([
            ('conv', 
             nn.Conv2d(
              mid_channels,
              mid_channels,
              kernel_size,
              stride,
              padding,
              groups=mid_channels,
              bias=bias)),
        ])
        if use_bn:
            depth_conv['bn'] = nn.BatchNorm2d(mid_channels, affine=affine, track_running_stats=affine)
        if act is not None:
            depth_conv['act'] = act(inplace=True)
        self.depth_conv = nn.Sequential(depth_conv)
        
        # se model
        if se_channels > 0:
            squeeze_excite = OrderedDict([
                    ('conv_reduce', nn.Conv2d(mid_channels, se_channels, 1, 1, 0, groups=groups, bias=True)),
                ])
            if act is not None:
                squeeze_excite['act'] = act(inplace=True)
            squeeze_excite['conv_expand'] = nn.Conv2d(se_channels, mid_channels, 1, 1, 0, groups=groups, bias=True)
            self.squeeze_excite = nn.Sequential(squeeze_excite)
        else:
            self.squeeze_excite = None
            self.se_channels = 0
        
        # pointwise linear
        point_linear = OrderedDict([
                ('conv', nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=bias)),
            ])
        if use_bn:
            point_linear['bn'] = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
        self.point_linear = nn.Sequential(point_linear)
        
        # residual flag
        self.has_residual = (in_channels == out_channels) and (stride == 1)
    
    def forward(self, x):
        res = x
        
        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
            if self.has_shuffle and self.groups > 1:
                x = channel_shuffle(x, self.groups)
        
        x = self.depth_conv(x)
        if self.squeeze_excite is not None:
            x_se = F.adaptive_avg_pool2d(x, 1)
            x = x * torch.sigmoid(self.squeeze_excite(x_se))
        
        x = self.point_linear(x)
        if self.has_shuffle and self.groups > 1:
            x = channel_shuffle(x, self.groups)
        
        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += res
        
        return x
    
    @property
    def name(self):
        return MBInvertedResBlock.__name__
    
    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            return '%dx%d_MBInvResBlock_E%.2f' % (kernel_size[0], kernel_size[1], 
                                      self.mid_channels * 1.0 / self.in_channels)
        else:
            return '%dx%d_GroupMBInvResBlock_E%.2f_G%d' % (kernel_size[0], kernel_size[1], 
    	                     self.mid_channels * 1.0 / self.in_channels, self.groups)
    
    @property
    def config(self):
        return {
            'name': MBInvertedResBlock.__name__,
            'in_channels': self.in_channels,
            'mid_channels': self.mid_channels,
            'se_channels': self.se_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'groups': self.groups,
            'has_shuffle': self.has_shuffle,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'affine': self.affine,
            'act_func': self.act_func,
        }
    
    @staticmethod
    def build_from_config(config):
        return MBInvertedResBlock(**config)
    
    
    def get_flops(self):
        raise NotImplementedError
    
    def get_latency(self):
        raise NotImplementedError



"""for steg pre-processing"""
import numpy as np

"""refer to SiaStegNet"""
SRM_npy = np.load(os.path.join(os.path.dirname(__file__), 'SRM_Kernels.npy'))
class SRMConv2d(nn.Module):

    def __init__(self, stride=1, padding=0, is_opt=True):
        super(SRMConv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        self.is_opt = is_opt
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = nn.Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=is_opt)
        if is_opt:
            self.bias = nn.Parameter(torch.Tensor(30), requires_grad=is_opt)
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)



