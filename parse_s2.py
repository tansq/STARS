import os
import sys
import argparse
import pickle
import json
import numpy as np
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models.model_eval_steg_op_depth import NetworkGC, PRIMITIVES, PRIMITIVES_stem
from tools.utils import count_parameters_in_MB
from tools.flops_benchmark import calculate_FLOPs_in_M


cudnn.enabled = True
cudnn.benchmark = True


def get_op_and_depth_weights(model_or_path):
    if isinstance(model_or_path, str):  # for model_path
        checkpoint = torch.load(model_or_path)
        state_dict = checkpoint['state_dict']
    else:                               # for model
        state_dict = model_or_path.state_dict()
    
    # stem_weights = []
    op_weights = []
    depth_weights = []
    
    for key in state_dict:
        if key.endswith('log_alphas'):
            op_weights.append(np.exp(state_dict[key].cpu().numpy()))
        elif key.endswith('betas'):
            depth_weights.append(F.softmax(state_dict[key].cpu(), dim=-1).numpy())
        else:
            continue

    print("log_alphas:")
    for i in op_weights:
        print(i)
    print("betas:")
    for i in depth_weights:
        print(i)
    
    return op_weights, depth_weights


def resume_weights(searched_model_path, model, parsed_stem=None):
    """ Resume model weights from search model
    searched_model_or_path
    cur_model: model to be resumed from searched model"""
    if isinstance(searched_model_path, str):
        searched_dict = torch.load(searched_model_path)['state_dict']
    else:
        searched_dict = searched_model_path.state_dict()
    
    model_path = os.path.join(os.path.dirname(searched_model_path), 'model_finetuned.pth.tar')
    
    # For cur_keys
    cur_keys = ['first_stem.conv.weight', 'first_stem.conv.bias', 'L4.conv1.conv.weight', 'L4.conv2.conv.weight',
    			 'L4.downsample.conv.weight', 'classifier.linear.weight', 'classifier.linear.bias']
    searched_keys = copy.deepcopy(cur_keys)
    
    parsed_arch = model.parsed_arch

    assert len(searched_keys) == len(cur_keys)
    for ckey, search_key in zip(cur_keys, searched_keys):
        assert search_key in searched_dict.keys(), "%s not existed in searched_dict!"%(search_key)
        assert ckey in model.state_dict().keys(), "%s not existed in model_dict!"%(ckey)
        with torch.no_grad():
            exec("model.{}.copy_(searched_dict[search_key])".format(ckey))

    for stage in parsed_arch:
        for i, block in enumerate(parsed_arch[stage]):
            op_idx = parsed_arch[stage][block]
            if PRIMITIVES[op_idx].startswith('BasicRes_'):
                # conv1
                searched_key = '{}.{}.m_ops.{}.conv1.conv.weight'.format(stage, block, op_idx)
                cur_key = "model.{}[{}].conv1.conv.weight".format(stage, i)
                with torch.no_grad():
                    exec(cur_key + ".copy_(searched_dict[searched_key])")
                searched_key = '{}.{}.m_ops.{}.conv2.conv.weight'.format(stage, block, op_idx)
                cur_key = "model.{}[{}].conv2.conv.weight".format(stage, i)
                with torch.no_grad():
                    exec(cur_key + ".copy_(searched_dict[searched_key])")
                # downsample 是否在该块中
                searched_key = '{}.{}.m_ops.{}.downsample.conv.weight'.format(stage, block, op_idx)
                if searched_key not in searched_dict.keys():
                    continue
                cur_key = "model.{}[{}].downsample.conv.weight".format(stage, i)
                with torch.no_grad():
                    # print(cur_key + ".copy_(searched_dict[searched_key])")
                    exec(cur_key + ".copy_(searched_dict[searched_key])")
            elif PRIMITIVES[op_idx].startswith('UCT3Res_'):
                # inverted_bottleneck
                searched_key = '{}.{}.m_ops.{}.inverted_bottleneck.conv.weight'.format(stage, block, op_idx)
                cur_key = "model.{}[{}].inverted_bottleneck.conv.weight".format(stage, i)
                with torch.no_grad():
                    exec(cur_key + ".copy_(searched_dict[searched_key])")
                # depth_conv
                searched_key = '{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
                cur_key = "model.{}[{}].depth_conv.conv.weight".format(stage, i)
                with torch.no_grad():
                    exec(cur_key + ".copy_(searched_dict[searched_key])")
                # point_linear
                searched_key = '{}.{}.m_ops.{}.point_linear.conv.weight'.format(stage, block, op_idx)
                cur_key = "model.{}[{}].point_linear.conv.weight".format(stage, i)
                with torch.no_grad():
                    exec(cur_key + ".copy_(searched_dict[searched_key])")
                # downsample
                searched_key = '{}.{}.m_ops.{}.downsample.conv.weight'.format(stage, block, op_idx)
                if searched_key not in searched_dict.keys():
                    continue
                cur_key = "model.{}[{}].downsample.conv.weight".format(stage, i)
                with torch.no_grad():
                    exec(cur_key + ".copy_(searched_dict[searched_key])")
            elif PRIMITIVES[op_idx].startswith('MBI_'):
                # inverted_bottleneck
                searched_key = '{}.{}.m_ops.{}.inverted_bottleneck.conv.weight'.format(stage, block, op_idx)
                if searched_key in searched_dict.keys():
                    cur_key = "model.{}[{}].inverted_bottleneck.conv.weight".format(stage, i)
                    with torch.no_grad():
                        exec(cur_key + ".copy_(searched_dict[searched_key])")
                # depth_conv
                searched_key = '{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
                cur_key = "model.{}[{}].depth_conv.conv.weight".format(stage, i)
                exec("print({}.size())".format(cur_key))
                exec("print(searched_dict[searched_key].size())")
                with torch.no_grad():
                    exec(cur_key + ".copy_(searched_dict[searched_key])")

                # squeeze_excite
                searched_key = '{}.{}.m_ops.{}.squeeze_excite.conv_reduce.weight'.format(stage, block, op_idx)
                if searched_key in searched_dict.keys():
                    cur_key = "model.{}[{}].squeeze_excite.conv_reduce.weight".format(stage, i)
                    with torch.no_grad():
                        exec(cur_key + ".copy_(searched_dict[searched_key])")
                    searched_key = '{}.{}.m_ops.{}.squeeze_excite.conv_reduce.bias'.format(stage, block, op_idx)
                    cur_key = "model.{}[{}].squeeze_excite.conv_reduce.bias".format(stage, i)
                    with torch.no_grad():
                        exec(cur_key + ".copy_(searched_dict[searched_key])")

                    searched_key = '{}.{}.m_ops.{}.squeeze_excite.conv_expand.weight'.format(stage, block, op_idx)
                    cur_key = "model.{}[{}].squeeze_excite.conv_expand.weight".format(stage, i)
                    with torch.no_grad():
                        exec(cur_key + ".copy_(searched_dict[searched_key])")
                    searched_key = '{}.{}.m_ops.{}.squeeze_excite.conv_expand.bias'.format(stage, block, op_idx)
                    cur_key = "model.{}[{}].squeeze_excite.conv_expand.bias".format(stage, i)
                    with torch.no_grad():
                        exec(cur_key + ".copy_(searched_dict[searched_key])")

                # point_linear
                searched_key = '{}.{}.m_ops.{}.point_linear.conv.weight'.format(stage, block, op_idx)
                cur_key = "model.{}[{}].point_linear.conv.weight".format(stage, i)
                with torch.no_grad():
                    exec(cur_key + ".copy_(searched_dict[searched_key])")

            else:
                raise NotImplementedError
    
    torch.save({
        'state_dict': model.state_dict(),
    }, model_path)

def parse_architecture(op_weights, depth_weights):
    
    parsed_arch = OrderedDict([
        ('stage1', OrderedDict([('block1', -1), ('block2', -1), ('block3', -1)])),
        ('stage2', OrderedDict([('block1', -1), ('block2', -1)])),
        ('stage3', OrderedDict([('block1', -1), ('block2', -1)])),
        ('stage4', OrderedDict([('block1', -1), ('block2', -1)])),
    ])
    
    start_res_dict = OrderedDict([
        ('stage1', 1),
        ('stage2', 0),
        ('stage3', 1),
        ('stage4', 1),
    ])
    
    stages = []
    blocks = []
    for stage in parsed_arch:
        for block in parsed_arch[stage]:
            stages.append(stage)
            blocks.append(block)
    
    op_max_indexes = [np.argmax(x) for x in op_weights]
    for (stage, block, op_max_index) in zip(stages, blocks, op_max_indexes):
        parsed_arch[stage][block] = op_max_index
    
    if len(depth_weights) > 0:
        assert len(depth_weights) == len(start_res_dict)
        depth_max_indexes = [np.argmax(x)+a for x, a in zip(depth_weights, start_res_dict.values())]
        for stage_index, depth_max_index in enumerate(depth_max_indexes, start=1):
            stage = 'stage{}'.format(stage_index)
            for block_index in range(depth_max_index+1, 5+1):
                block = 'block{}'.format(block_index)
                if block in parsed_arch[stage]:
                    del parsed_arch[stage][block]
    
    print(parsed_arch)
    return parsed_arch




if __name__ == '__main__':
    parser = argparse.ArgumentParser("parsing TF-NAS")
    parser.add_argument('--model_path', type=str, required=True, help='path of searched model')
    parser.add_argument('--save_model', type=str, default='.', help='saving path of parsed architecture')
    parser.add_argument('--save_cfg', type=str, default='.', help='saving path of parsed architecture config')
    
    args = parser.parse_args()
    
    op_weights, depth_weights = get_op_and_depth_weights(args.model_path)
    parsed_arch = parse_architecture(op_weights, depth_weights)
    model = NetworkGC(2, 3, parsed_arch, 0.0)

    parsed_stem = None
    model = model.cuda()
    
    x = torch.randn((1, 3, 256, 256))
    x = x.cuda()
    
    config = model.config
    with open(args.save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    params = count_parameters_in_MB(model)
    print('Params: {:.4f}MB'.format(params))
    
    flops = calculate_FLOPs_in_M(model, (1, 3, 256, 256))
    print('FLOPs: {:.4f}M'.format(flops))
    
    resume_weights(args.model_path, model, parsed_stem)
