import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
from .model_search_steg_op_depth import PRIMITIVES_stem, PRIMITIVES, OPS


class NetworkGC(nn.Module):
    def __init__(self, num_classes, img_chs, parsed_arch, dropout_rate=0.0):
        super(NetworkGC, self).__init__()
        self.parsed_arch = parsed_arch
        self.dropout_rate = dropout_rate
        self.block_idx = 0

        self.first_stem = ConvLayer(img_chs, 90, kernel_size=5, stride=1, groups=3,
                                    affine=True, act_func='relu', bias=True)

        self.stage1 = self._make_stage('stage1',
            ics=[90, 30, 30],
            ocs=[30, 30, 30],
            ss=[1, 1, 1],
            affs=[True, True, True],
            acts=['relu', 'relu', 'relu'], )
        self.L4 = BasicResBlock(30, 0, 64, stride=2, affine=True)
        self.stage2 = self._make_stage('stage2',
            ics=[64, 64],
            ocs=[64, 64],
            ss=[1, 1],
            affs=[True, True],
            acts=['relu', 'relu'], )
        self.stage3 = self._make_stage('stage3',
            ics=[64, 128],
            ocs=[128, 128],
            ss=[2, 1],
            affs=[True, True],
            acts=['relu', 'relu'], )
        self.stage4 = self._make_stage('stage4',
            ics=[128, 256],
            ocs=[256, 256],
            ss=[2, 1],
            affs=[True, True],
            acts=['relu', 'relu'], )

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = LinearLayer(256, num_classes)

        self._initialization()

    def _make_stage(self, stage_name, ics, ocs, ss, affs, acts):
        stage = nn.ModuleList()

        if self.parsed_arch[stage_name] == OrderedDict():
            return stage
        for i, block_name in enumerate(self.parsed_arch[stage_name]):
            self.block_idx += 1
            op_idx = self.parsed_arch[stage_name][block_name]
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

    def forward(self, x):
        x = self.first_stem(x)

        for block in self.stage1:
            x = block(x)
        x = self.L4(x)
        for block in self.stage2:
            x = block(x)
        for block in self.stage3:
            x = block(x)
        for block in self.stage4:
            x = block(x)

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        return x

    @property
    def config(self):
        return {
            'first_stem': self.first_stem.config,
            'stage1': [block.config for block in self.stage1],
            'L4': self.L4.config,
            'stage2': [block.config for block in self.stage2],
            'stage3': [block.config for block in self.stage3],
            'stage4': [block.config for block in self.stage4],
            'classifier': self.classifier.config,
        }

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


class NetworkGCCfg(nn.Module):
    def __init__(self, num_classes, img_chs, model_config, dropout_rate=0.0):
        super(NetworkGCCfg, self).__init__()
        self.model_config = model_config
        self.dropout_rate = dropout_rate
        self.block_idx = 0

        self.first_stem = set_layer_from_config(model_config['first_stem'])

        self.stage1 = self._make_stage('stage1')
        self.L4 = set_layer_from_config(model_config['L4'])
        self.stage2 = self._make_stage('stage2')
        self.stage3 = self._make_stage('stage3')
        self.stage4 = self._make_stage('stage4')
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        classifier_config = model_config['classifier']
        self.classifier = set_layer_from_config(classifier_config)

        self._initialization()

    def _make_stage(self, stage_name):
        stage = nn.ModuleList()
        for layer_config in self.model_config[stage_name]:
            self.block_idx += 1
            layer = set_layer_from_config(layer_config)
            stage.append(layer)

        return stage

    def forward(self, x):
        x = self.first_stem(x)

        for block in self.stage1:
            x = block(x)
        x = self.L4(x)
        for block in self.stage2:
            x = block(x)
        for block in self.stage3:
            x = block(x)
        for block in self.stage4:
            x = block(x)

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        return x

    @property
    def config(self):
        return {
            'first_stem': self.first_stem.config,
            'stage1': [block.config for block in self.stage1],
            'L4': self.L4.config,
            'stage2': [block.config for block in self.stage2],
            'stage3': [block.config for block in self.stage3],
            'stage4': [block.config for block in self.stage4],
            'classifier': self.classifier.config,
        }

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

