
import os
import argparse
import numpy as np
import json
from collections import OrderedDict

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from models.model_search_steg_op_depth import NAS_GroupConv, PRIMITIVES_stem, PRIMITIVES


cudnn.enabled = True
cudnn.benchmark = True


def get_op_and_depth_weights(model_or_path):
    if isinstance(model_or_path, str):  # for model_path
        checkpoint = torch.load(model_or_path)
        state_dict = checkpoint['state_dict']
    else:  # for model
        state_dict = model_or_path.state_dict()

    op_weights = []
    depth_weights = []

    for key in state_dict:
        if key.endswith('stem_log_alphas') or key.endswith('log_alphas'):
            op_weights.append(np.exp(state_dict[key].cpu().numpy()))
        elif key.endswith('stem_betas'):
            depth_weights.append(F.softmax(state_dict[key].cpu(), dim=-1).numpy())
        else:
            continue

    print("stem_log_alphas:")
    for i in op_weights:
        print(i)
    print("stem_betas:")
    for i in depth_weights:
        print(i)

    return op_weights, depth_weights


def parse_architecture(model_or_path):
    if isinstance(model_or_path, str):  # for model_path
        checkpoint = torch.load(model_or_path, map_location='cpu')
        searched_dict = checkpoint['state_dict']
    elif isinstance(model_or_path, list):
        return [np.argmax(x) for x in model_or_path]
    else:                               # for model
        searched_dict = model_or_path.state_dict()

    # get stem_op name
    op_weights = []
    depth_weights = []

    for key in searched_dict:
        if key.endswith('stem_log_alphas') or key.endswith('log_alphas'):
            op_weights.append(np.exp(searched_dict[key].cpu().numpy()))
        elif key.endswith('stem_betas') or key.endswith('betas'):
            depth_weights.append(F.softmax(searched_dict[key].cpu(), dim=-1).numpy())
        else:
            continue

    parsed_arch = OrderedDict([
        ('first_stem', OrderedDict([('block1', -1)])),
    ])

    stems = []
    blocks = []
    for stem in parsed_arch:
        for block in parsed_arch[stem]:
            stems.append(stem)
            blocks.append(block)

    # parse_arch = [np.argmax(x) for x in op_weights]
    op_max_indexes = [np.argmax(x) for x in op_weights]
    for (stem, block, op_max_index) in zip(stems, blocks, op_max_indexes):
        parsed_arch[stem][block] = op_max_index	# 选择最好的op_idx

    # stem = 'second_stem'
    if len(depth_weights) > 0:
        depth_max_indexes = [np.argmax(x) + 1 for x in zip(depth_weights)]
        for block_index in range(depth_max_indexes[0] + 1, 5 + 1):  # 跳过该阶段depth_max_index后面的block
            block = 'block{}'.format(block_index)
            if block in parsed_arch['second_stem']:
                del parsed_arch['second_stem'][block]

    return parsed_arch

def resume_stem_weights(model_or_path, model, save_path):
    if isinstance(model_or_path, str):  # for model_path
        checkpoint = torch.load(model_or_path, map_location='cpu')
        searched_dict = checkpoint['state_dict']
    else:                               # for model
        searched_dict = model_or_path.searched_dict()

    # print(model.state_dict()["second_stem.0.depth_conv.conv.weight"])
    # print(model.second_stem[0])

    # get stem_op name
    # op_weights = []
    # for key in searched_dict:
    #     if key.endswith('stem_log_alphas') or key.endswith('log_alphas'):
    #         op_weights.append(np.exp(searched_dict[key].cpu().numpy()))
    #     else:
    #         continue
    #
    # print("stem_log_alphas:")
    # for i in op_weights:
    #     print(i)
    # # assert len(op_weights) == 1
    # op_max_indexes = [np.argmax(x) for x in op_weights]

    parsed_arch = model.parsed_arch

    searched_keys = []
    cur_keys = []

    for stage in parsed_arch:
        if stage == 'first_stem':
            op_idx = parsed_arch[stage]['block1']
            op_name = PRIMITIVES_stem[op_idx]
            print(op_name)
            searched_keys.append("first_stem.m_ops.{}.conv.weight".format(op_idx))
            searched_keys.append("first_stem.m_ops.{}.conv.bias".format(op_idx))
            cur_keys.append("first_stem.conv.weight")
            cur_keys.append("first_stem.conv.bias")
        
        else:
            raise NotImplementedError


    for ckey, search_key in zip(cur_keys, searched_keys):
        assert search_key in searched_dict.keys(), "%s not existed in searched_dict!"%(search_key)
        # assert ckey in model.state_dict().keys(), "%s not existed in model_dict!"%(ckey)
        with torch.no_grad():
            exec("model.{}.copy_(searched_dict[search_key])".format(ckey))

    torch.save({
        'state_dict': model.state_dict(),
        'epoch': -1,
    }, save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("parsing model from stage1 to stage2")
    parser.add_argument('--model_path', type=str, required=True, help='path of searched model')
    parser.add_argument('--save_path', type=str, default='', help='saving path of parsed architecture config')
    args = parser.parse_args()

    if args.save_path == "":
        os.path.join(os.path.dirname(args.model_path), "stage2_resume_stem.pth.tar")

    parsed_arch = parse_architecture(args.model_path)
    model = NAS_GroupConv(num_classes=2, img_chs=3, parsed_arch=parsed_arch, p=0.0)
    print(model.state_dict())

    config = model.config
    save_cfg = os.path.join(os.path.dirname(args.save_path), 'stem.config')
    with open(save_cfg, 'w') as f:
        json.dump(config, f, indent=4)

    op_weights = resume_stem_weights(args.model_path, model, args.save_path)
