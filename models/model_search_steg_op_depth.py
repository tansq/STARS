import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


PRIMITIVES_stem = [
    'SepConv_k3',
    'SepConv_k5',
    'SepConv_k7',
    'Conv_k3g3',
    'Conv_k5g3',
    'Conv_k7g3',
    'Conv_k3',
    'Conv_k5',
    'Conv_k7',
]


PRIMITIVES = [
    'BasicRes_k3',
    'MBI_k3_e0',
    'MBI_k5_e0',
    'MBI_k3_e3',
    'MBI_k5_e3',
]

OPS = {
    'BasicRes_k3': lambda ic, mc, oc, s, aff, act: BasicResBlock(ic, 0, oc, 3, s, affine=aff, act_func=act),
    'BasicRes_k5': lambda ic, mc, oc, s, aff, act: BasicResBlock(ic, 0, oc, 5, s, affine=aff, act_func=act),
    'UCT3Res_k3': lambda ic, mc, oc, s, aff, act: UCT3ResBlock(ic, 0, oc, 3, s, affine=aff, act_func=act),
    'SRRes_k3': lambda ic, mc, oc, s, aff, act: SRBlock(ic, 0, oc, 3, s, affine=aff, act_func=act),
    'MBI_k3_e0' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, 0, 0, oc, 3, s, affine=aff, act_func=act),
    'MBI_k5_e0' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, 0, 0, oc, 5, s, affine=aff, act_func=act),
    'MBI_k3_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
    'MBI_k5_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
    'MBI_k3_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 3, s, affine=aff, act_func=act),
    'MBI_k5_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 5, s, affine=aff, act_func=act),
    # 'skip'      : lambda ic, mc, oc, s, aff, act: IdentityLayer(ic, oc),
    'Conv_k3': lambda ic, oc, s, aff, act, bias: ConvLayer(ic, oc, 3, s, 1, affine=aff, act_func=act, bias=bias),
    'Conv_k5': lambda ic, oc, s, aff, act, bias: ConvLayer(ic, oc, 5, s, 1, affine=aff, act_func=act, bias=bias),
    'Conv_k7': lambda ic, oc, s, aff, act, bias: ConvLayer(ic, oc, 7, s, 1, affine=aff, act_func=act, bias=bias),
    'Conv_k3g3': lambda ic, oc, s, aff, act, bias: ConvLayer(ic, oc, 3, s, 3, affine=aff, act_func=act, bias=bias),
    'Conv_k5g3': lambda ic, oc, s, aff, act, bias: ConvLayer(ic, oc, 5, s, 3, affine=aff, act_func=act, bias=bias),
    'Conv_k7g3': lambda ic, oc, s, aff, act, bias: ConvLayer(ic, oc, 7, s, 3, affine=aff, act_func=act, bias=bias),
    'SepConv_k3': lambda ic, oc, s, aff, act, bias: SepConvLayer(ic, oc, 3, s, affine=aff, act_func=act, bias=bias),
    'SepConv_k5': lambda ic, oc, s, aff, act, bias: SepConvLayer(ic, oc, 5, s, affine=aff, act_func=act, bias=bias),
    'SepConv_k7': lambda ic, oc, s, aff, act, bias: SepConvLayer(ic, oc, 7, s, affine=aff, act_func=act, bias=bias),
}


class MixedOP_stem(nn.Module):
    def __init__(self, in_channels, out_channels, stride, affine, act_func, bias, num_ops):
        super(MixedOP_stem, self).__init__()
        self.num_ops = num_ops
        self.m_ops = nn.ModuleList()

        for i in range(num_ops):
            primitive = PRIMITIVES_stem[i]
            op = OPS[primitive](in_channels, out_channels, stride, affine, act_func, bias)
            self.m_ops.append(op)

        self._initialize_log_alphas()
        self.reset_switches()

    def fink_ori_idx(self, idx):
        count = 0
        for ori_idx in range(len(self.switches)):
            if self.switches[ori_idx]:
                count += 1
                if count == (idx + 1):
                    break
        return ori_idx

    def forward(self, x, sampling, mode):
        if sampling:
            weights = self.stem_log_alphas[self.switches]  # 取 True对应的参数，维度与True个数相等
            if mode == 'gumbel':  # 一次采样
                weights = F.gumbel_softmax(F.log_softmax(weights, dim=-1), self.T, hard=False)
                idx = torch.argmax(weights).item()
                self.switches[idx] = False
            elif mode == 'gumbel_2':  # 二次采样时调用
                weights = F.gumbel_softmax(F.log_softmax(weights, dim=-1), self.T, hard=False)
                idx = torch.argmax(weights).item()
                idx = self.fink_ori_idx(idx)
                self.reset_switches()
            elif mode == 'min_alphas':
                idx = torch.argmin(weights).item()
                idx = self.fink_ori_idx(idx)
                self.reset_switches()
            elif mode == 'max_alphas':
                idx = torch.argmax(weights).item()
                idx = self.fink_ori_idx(idx)
                self.reset_switches()
            elif mode == 'random':
                idx = random.choice(range(len(weights)))
                idx = self.fink_ori_idx(idx)
                self.reset_switches()
            else:
                raise ValueError('invalid sampling mode...')
            op = self.m_ops[idx]
            return op(x)
        else:
            weights = F.gumbel_softmax(self.stem_log_alphas, self.T, hard=False)
            out = sum(w * op(x) for w, op in zip(weights, self.m_ops))
            return out

    def _initialize_log_alphas(self):
        alphas = torch.zeros((self.num_ops,))  # arch_params of candidate ops
        stem_log_alphas = F.log_softmax(alphas, dim=-1)
        self.register_parameter('stem_log_alphas', nn.Parameter(stem_log_alphas))

    def reset_switches(self):
        self.switches = [True] * self.num_ops  # switches，用于控制下一轮OP的选择范围

    def set_temperature(self, T):
        self.T = T


class MixedStage_stem(nn.Module):
    def __init__(self, ics, ocs, ss, affs, acts, bias, stage_type=1):
        super(MixedStage_stem, self).__init__()
        self.stage_type = stage_type  # 0 for stage6 || 1 for stage1 || 2 for stage2 || 3 for stage3/4/5 用于控制每一阶段的深度
        self.start_res = 0 if ((ics[0] == ocs[0]) and (ss[0] == 1)) else 1  # 若首层不改变通道数和s=1，则起始残差设为0，也即可能跳过该stage
        self.num_res = len(ics) - self.start_res + 1  # 残差求和的分支个数，也即beta的个数

        # stage6
        if stage_type == 0:
            self.block1 = MixedOP_stem(ics[0], ocs[0], ss[0], affs[0], acts[0], bias, len(PRIMITIVES_stem))
        # stage1
        elif stage_type == 1:
            self.block1 = MixedOP_stem(ics[0], ocs[0], ss[0], affs[0], acts[0], bias, len(PRIMITIVES_stem))
            self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES))
        # stage2
        elif stage_type == 2:
            self.block1 = MixedOP_stem(ics[0], ocs[0], ss[0], affs[0], acts[0], bias, len(PRIMITIVES_stem))
            self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES))
            self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES))
        # stage3, stage4, stage5
        elif stage_type == 3:
            self.block1 = MixedOP_stem(ics[0], ocs[0], ss[0], affs[0], acts[0], bias, len(PRIMITIVES_stem))
            self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES))
            self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES))
            self.block4 = MixedOP(ics[3], ocs[3], ss[3], affs[3], acts[3], len(PRIMITIVES))
        else:
            raise ValueError('invalid stage_type...')

        self._initialize_betas()

    def forward(self, x, sampling, mode):
        res_list = [x, ]

        # stage6
        if self.stage_type == 0:
            out1 = self.block1(x, sampling, mode)
            res_list.append(out1)
        # stage1
        elif self.stage_type == 1:
            out1 = self.block1(x, sampling, mode)
            res_list.append(out1)
            out2 = self.block2(out1, sampling, mode)
            res_list.append(out2)
        # stage2
        elif self.stage_type == 2:
            out1 = self.block1(x, sampling, mode)
            res_list.append(out1)
            out2 = self.block2(out1, sampling, mode)
            res_list.append(out2)
            out3 = self.block3(out2, sampling, mode)
            res_list.append(out3)
        # stage3, stage4, stage5
        elif self.stage_type == 3:
            out1 = self.block1(x, sampling, mode)
            res_list.append(out1)
            out2 = self.block2(out1, sampling, mode)
            res_list.append(out2)
            out3 = self.block3(out2, sampling, mode)
            res_list.append(out3)
            out4 = self.block4(out3, sampling, mode)
            res_list.append(out4)
        else:
            raise ValueError

        weights = F.softmax(self.stem_betas, dim=-1)
        out = sum(w * res for w, res in zip(weights, res_list[self.start_res:]))

        return out

    def _initialize_betas(self):	# 将beta参数加入该Module，用于控制深度搜索, arch_params of sink connections
        stem_betas = torch.zeros((self.num_res))
        self.register_parameter('stem_betas', nn.Parameter(stem_betas))


class MixedOP(nn.Module):
    def __init__(self, in_channels, out_channels, stride, affine, act_func, num_ops):
        super(MixedOP, self).__init__()
        self.num_ops = num_ops
        self.m_ops = nn.ModuleList()
        
        for i in range(num_ops):
            primitive = PRIMITIVES[i]
            if 'e3' in primitive:
                mid_channels = in_channels * 3
            elif 'e6' in primitive:
                mid_channels = in_channels * 6
            else:
                mid_channels = 0
            op = OPS[primitive](in_channels, mid_channels, out_channels, stride, affine, act_func)
            self.m_ops.append(op)
        
        self._initialize_log_alphas()
        self.reset_switches()

    def fink_ori_idx(self, idx):
        count = 0
        for ori_idx in range(len(self.switches)):
            if self.switches[ori_idx]:
                count += 1
                if count == (idx + 1):
                    break
        return ori_idx

    def forward(self, x, sampling, mode):
        if sampling:
            weights = self.log_alphas[self.switches]		# 取 True对应的参数，维度与True个数相等
            if mode == 'gumbel':	# 一次采样
                weights = F.gumbel_softmax(F.log_softmax(weights, dim=-1), self.T, hard=False)
                idx = torch.argmax(weights).item()
                self.switches[idx] = False
            elif mode == 'gumbel_2':	# 二次采样时调用
                weights = F.gumbel_softmax(F.log_softmax(weights, dim=-1), self.T, hard=False)
                idx = torch.argmax(weights).item()
                idx = self.fink_ori_idx(idx)
                self.reset_switches()
            elif mode == 'min_alphas':
                idx = torch.argmin(weights).item()
                idx = self.fink_ori_idx(idx)
                self.reset_switches()
            elif mode == 'max_alphas':
                idx = torch.argmax(weights).item()
                idx = self.fink_ori_idx(idx)
                self.reset_switches()
            elif mode == 'random':
                idx = random.choice(range(len(weights)))
                idx = self.fink_ori_idx(idx)
                self.reset_switches()
            else:
                raise ValueError('invalid sampling mode...')
            op = self.m_ops[idx]
            return op(x)
        else:
            weights = F.gumbel_softmax(self.log_alphas, self.T, hard=False)
            out = sum(w*op(x) for w, op in zip(weights, self.m_ops))

            return out

    def _initialize_log_alphas(self):
        alphas = torch.zeros((self.num_ops,))	# arch_params of candidate ops
        log_alphas = F.log_softmax(alphas, dim=-1)
        self.register_parameter('log_alphas', nn.Parameter(log_alphas))

    def reset_switches(self):
        self.switches = [True] * self.num_ops	# switches，用于控制下一轮OP的选择范围
    
    def set_temperature(self, T):
        self.T = T


class MixedStage(nn.Module):
    def __init__(self, ics, ocs, ss, affs, acts, stage_type=1):
        super(MixedStage, self).__init__()
        self.stage_type = stage_type # 0 for stage6 || 1 for stage1 || 2 for stage2 || 3 for stage3/4/5 用于控制每一阶段的深度
        self.start_res = 0 if ((ics[0] == ocs[0]) and (ss[0] == 1)) else 1	# 若首层不改变通道数和s=1，则起始残差设为0，也即可能跳过该stage
        self.num_res = len(ics) - self.start_res + 1	# 残差求和的分支个数，也即beta的个数
    
        # stage6
        if stage_type == 0:
            self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES))
        # stage1
        elif stage_type == 1:
            self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES))
            self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES))
        # stage2
        elif stage_type == 2:
            self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES))
            self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES))
            self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES))
        # stage3, stage4, stage5
        elif stage_type == 3:
            self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES))
            self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES))
            self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES))
            self.block4 = MixedOP(ics[3], ocs[3], ss[3], affs[3], acts[3], len(PRIMITIVES))
        else:
            raise ValueError('invalid stage_type...')
        
        self._initialize_betas()

    def forward(self, x, sampling, mode):
        res_list = [x,]
        
        # stage6
        if self.stage_type == 0:
            out1 = self.block1(x, sampling, mode)
            res_list.append(out1)
        # stage1
        elif self.stage_type == 1:
            out1 = self.block1(x, sampling, mode)
            res_list.append(out1)
            out2 = self.block2(out1, sampling, mode)
            res_list.append(out2)
        # stage2
        elif self.stage_type == 2:
            out1 = self.block1(x, sampling, mode)
            res_list.append(out1)
            out2 = self.block2(out1, sampling, mode)
            res_list.append(out2)
            out3 = self.block3(out2, sampling, mode)
            res_list.append(out3)
        # stage3, stage4, stage5
        elif self.stage_type == 3:
            out1 = self.block1(x, sampling, mode)
            res_list.append(out1)
            out2 = self.block2(out1, sampling, mode)
            res_list.append(out2)
            out3 = self.block3(out2, sampling, mode)
            res_list.append(out3)
            out4 = self.block4(out3, sampling, mode)
            res_list.append(out4)
        else:
            raise ValueError
        
        weights = F.softmax(self.betas, dim=-1)
        out = sum(w*res for w, res in zip(weights, res_list[self.start_res:]))
        
        return out

    def _initialize_betas(self):	# 将beta参数加入该Module，用于控制深度搜索, arch_params of sink connections
        betas = torch.zeros((self.num_res))
        self.register_parameter('betas', nn.Parameter(betas))


class MixedStage_fix(nn.Module):
    def __init__(self, ics, ocs, ss, affs, acts, stage_type=1):
        super(MixedStage_fix, self).__init__()
        self.stage_type = stage_type  # 0 for stage6 || 1 for stage1 || 2 for stage2 || 3 for stage3/4/5 用于控制每一阶段的深度
        self.start_res = 0 if ((ics[0] == ocs[0]) and (ss[0] == 1)) else 1  # 若首层不改变通道数和s=1，则起始残差设为0，也即可能跳过该stage
        self.num_res = len(ics) - self.start_res + 1  # 残差求和的分支个数，也即beta的个数

        # stage6
        if stage_type == 0:
            self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES))
        # stage1
        elif stage_type == 1:
            self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES))
            self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES))
        # stage2
        elif stage_type == 2:
            self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES))
            self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES))
            self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES))
        # stage3, stage4, stage5
        elif stage_type == 3:
            self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES))
            self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES))
            self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES))
            self.block4 = MixedOP(ics[3], ocs[3], ss[3], affs[3], acts[3], len(PRIMITIVES))
        else:
            raise ValueError('invalid stage_type...')


    def forward(self, x, sampling, mode):
        res_list = [x, ]

        # stage6
        if self.stage_type == 0:
            out1 = self.block1(x, sampling, mode)
            # res_list.append(out1)
            return out1
        # stage1
        elif self.stage_type == 1:
            out1 = self.block1(x, sampling, mode)
            # res_list.append(out1)
            out2 = self.block2(out1, sampling, mode)
            # res_list.append(out2)
            return out2
        # stage2
        elif self.stage_type == 2:
            out1 = self.block1(x, sampling, mode)
            # res_list.append(out1)
            out2 = self.block2(out1, sampling, mode)
            # res_list.append(out2)
            out3 = self.block3(out2, sampling, mode)
            # res_list.append(out3)
            return out3
        # stage3, stage4, stage5
        elif self.stage_type == 3:
            out1 = self.block1(x, sampling, mode)
            # res_list.append(out1)
            out2 = self.block2(out1, sampling, mode)
            # res_list.append(out2)
            out3 = self.block3(out2, sampling, mode)
            # res_list.append(out3)
            out4 = self.block4(out3, sampling, mode)
            # res_list.append(out4)
            return out4
        else:
            raise ValueError


class NAS_GroupConv(nn.Module):

    def __init__(self, num_classes=2, img_chs=3, parsed_arch=None, p=0.0):
        super(NAS_GroupConv, self).__init__()

        self.parsed_arch = parsed_arch
        affine = False
        self.block_idx = 0

        self.first_stem = ConvLayer(img_chs, 90, kernel_size=5, stride=1, groups=3,
                                    affine=affine, act_func='relu', bias=True)

        self.stage1 = MixedStage(
            ics=[90, 30, 30],
            ocs=[30, 30, 30],
            ss=[1, 1, 1],
            affs=[False, False, False],
            acts=['relu', 'relu', 'relu'],
            stage_type=2, )
        self.L4 = BasicResBlock(30, 0, 64, stride=2, affine=affine)
        self.stage2 = MixedStage(
            ics=[64, 64],
            ocs=[64, 64],
            ss=[1, 1],
            affs=[False, False],
            acts=['relu', 'relu'],
            stage_type=1, )
        self.stage3 = MixedStage(
            ics=[64, 128],
            ocs=[128, 128],
            ss=[2, 1],
            affs=[False, False],
            acts=['relu', 'relu'],
            stage_type=1, )
        self.stage4 = MixedStage(
            ics=[128, 256],
            ocs=[256, 256],
            ss=[2, 1],
            affs=[False, False],
            acts=['relu', 'relu'],
            stage_type=1, )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = LinearLayer(256, num_classes)
        self.dropout = nn.Dropout(p=p)

        self._initialization()


    def _make_stem(self, stage_name, ics, ocs, ss, affs, acts, bias):
        stage = nn.ModuleList()

        # 检查该stage，当 (ics[0]==ocs[0] and ss[0]) 成立时，去掉该stage_name 的最后一个block
        # if ((ics[0] == ocs[0]) and (ss[0] == 1)):	# 不是所有都这么做，如果参数对应最后一个block呢？
        # 	self.parsed_arch[stage_name].popitem()
        if self.parsed_arch[stage_name] == OrderedDict():
            return stage
        for i, block_name in enumerate(self.parsed_arch[stage_name]):
            self.block_idx += 1
            op_idx = self.parsed_arch[stage_name][block_name]
            if i == 0:
                primitive = PRIMITIVES_stem[op_idx]
                op = OPS[primitive](ics[i], ocs[i], ss[i], affs[i], acts[i], bias)
            else:
                primitive = PRIMITIVES[op_idx]
                if 'e3' in primitive:
                    mc = ics[i] * 3
                elif 'e6' in primitive:
                    mc = ics[i] * 6
                else:
                    mc = 0
                op = OPS[primitive](ics[i], mc, ocs[i], ss[i], affs[i], acts[i])
            stage.append(op)

        return stage

    def forward(self, inputs, sampling, mode='max'):
        out = self.first_stem(inputs)

        out = self.stage1(out, sampling, mode)
        out = self.L4(out)
        out = self.stage2(out, sampling, mode)
        out = self.stage3(out, sampling, mode)
        out = self.stage4(out, sampling, mode)

        out = self.avgpool(out)
        out = out.view(out.size(0), out.size(1))

        out = self.dropout(out)
        out = self.classifier(out)

        return out

    def _initialization(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.__class__.__name__ != 'SRMConv2d':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

    @property
    def config(self):
        return {
            'first_stem': self.first_stem.config,
        }

    def weight_parameters(self):
        _weight_parameters = []

        for k, v in self.named_parameters():
            if not (k.endswith('log_alphas') or k.endswith('betas')):  # or k.endswith('hpf.weight')):
                _weight_parameters.append(v)

        return _weight_parameters

    def arch_parameters(self):
        _arch_parameters = []

        for k, v in self.named_parameters():
            if k.endswith('log_alphas') or k.endswith('betas'):
                _arch_parameters.append(v)

        return _arch_parameters

    def log_alphas_parameters(self):
        _log_alphas_parameters = []

        for k, v in self.named_parameters():
            if k.endswith('log_alphas'):
                _log_alphas_parameters.append(v)

        return _log_alphas_parameters

    def betas_parameters(self):
        _betas_parameters = []

        for k, v in self.named_parameters():
            if k.endswith('betas'):
                _betas_parameters.append(v)

        return _betas_parameters

    def set_temperature(self, T):
        for m in self.modules():
            if isinstance(m, MixedOP):
                m.set_temperature(T)

    def reset_switches(self):
        for m in self.modules():
            if isinstance(m, MixedOP):
                m.reset_switches()


class NAS_GroupConv_stage1(nn.Module):

    def __init__(self, num_classes=2, img_chs=3, p=0.0):
        super(NAS_GroupConv_stage1, self).__init__()

        affine = False

        self.first_stem = MixedOP_stem(img_chs, 90, 1, affine=affine, act_func='relu', bias=True, num_ops=len(PRIMITIVES_stem))

        self.L1 = BasicResBlock(90, 0, 30, 3, affine=affine)
        self.L2 = BasicResBlock(30, 0, 64, 3, stride=2, affine=affine)
        self.L3 = BasicResBlock(64, 0, 128, 3, stride=2, affine=affine)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = LinearLayer(128, num_classes)
        # self.dropout = nn.Dropout(p=p)

        self._initialization()


    def forward(self, inputs, sampling, mode='max'):
        out_features = self.first_stem(inputs, sampling, mode)

        out = self.L1(out_features)
        out = self.L2(out)
        out = self.L3(out)

        out = self.global_avg_pooling(out)
        out = out.view(out.size(0), out.size(1))
        # out = self.dropout(out)
        out = self.classifier(out)

        return out, out_features

    def _initialization(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.__class__.__name__ != 'SRMConv2d':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

    @property
    def config(self):
        return {
            'first_stem': self.first_stem.config,
        }

    def weight_parameters(self):
        _weight_parameters = []

        for k, v in self.named_parameters():
            if not (k.endswith('stem_log_alphas') or k.endswith('stem_betas')):  # or k.endswith('hpf.weight')):
                _weight_parameters.append(v)

        return _weight_parameters

    def arch_parameters(self):
        _arch_parameters = []

        for k, v in self.named_parameters():
            if k.endswith('stem_log_alphas') or k.endswith('stem_betas'):
                _arch_parameters.append(v)

        return _arch_parameters

    def log_alphas_parameters(self):
        _log_alphas_parameters = []

        for k, v in self.named_parameters():
            if k.endswith('stem_log_alphas'):
                _log_alphas_parameters.append(v)

        return _log_alphas_parameters

    def betas_parameters(self):
        _betas_parameters = []

        for k, v in self.named_parameters():
            if k.endswith('stem_betas'):
                _betas_parameters.append(v)

        return _betas_parameters

    def set_temperature(self, T):
        for m in self.modules():
            if isinstance(m, MixedOP_stem):
                m.set_temperature(T)

    def reset_switches(self):
        for m in self.modules():
            if isinstance(m, MixedOP_stem):
                m.reset_switches()
