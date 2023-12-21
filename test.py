import os
import sys
import time
import glob
import logging
import argparse
import json
import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tools.utils import AverageMeter, accuracy_2cls, alaska_weighted_auc, tf_roc
from tools.utils import count_parameters_in_MB
from tools.flops_benchmark import calculate_FLOPs_in_M
from models.model_eval_steg_op_depth import NetworkGC, NetworkGCCfg

from parsing_model import get_op_and_depth_weights
from parsing_model import parse_architecture
from dataset.stegdatas import CustomDataset_color


parser = argparse.ArgumentParser("testing the trained architectures")
# various path

parser.add_argument('--cover_dir', type=str, default='../cover_data', help='location of the cover_data corpus')
parser.add_argument('--stego_dir', type=str, default='../stego_data', help='location of the stego_data corpus')
parser.add_argument('--list_dir', type=str, default='.', help='data list folder like bb/train.txt')
parser.add_argument('--img_chs', type=int, default=3, help='input channels number')
parser.add_argument('--num_classes', type=int, default=2, help='class number of training set')
parser.add_argument('--image_size', type=int, default=256, help='image size of training set')
# parser.add_argument('--normalize', type=int, default=0, choices=[0, 1], help='Normalize images')
# parser.add_argument('--aug', type=int, default=1, choices=[0, 1], help='augment dataset: flip and rot')
# parser.add_argument('--img_type', type=str, default=None, help='non-rounded ycrcb(default) or rgb')
# parser.add_argument('--data_load', type=str, default='mydata', choices=['mydata', 'custom'], help='class to read data')

parser.add_argument('--model_path', type=str, default='', help='the searched model path')
parser.add_argument('--config_path', type=str, default='', help='the model config path')
parser.add_argument('--weights', type=str, required=True, help='pretrained model weights')
# parser.add_argument('--model_name', type=str, default='derived', help='pretrained model weights')

# training hyper-parameters
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

args, unparsed = parser.parse_known_args()

assert os.path.isfile(args.weights)
args.save = os.path.dirname(args.weights)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'test_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
    cudnn.enabled=True
    cudnn.benchmark = True
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    
    # create model
    # if args.model_name == 'derived':
    if args.model_path and os.path.isfile(args.model_path):
        op_weights, depth_weights = get_op_and_depth_weights(args.model_path)
        parsed_arch = parse_architecture(op_weights, depth_weights)
        model = NetworkGC(args.num_classes, args.img_chs, parsed_arch, 0.0)
    elif args.config_path and os.path.isfile(args.config_path):
        model_config = json.load(open(args.config_path, 'r'))
        model = NetworkGCCfg(args.num_classes, args.img_chs, model_config, 0.0)
    else:
        raise Exception('invalid --model_path and --config_path')
    
    # load pretrained weights
    if os.path.exists(args.weights):
        logging.info('loading weights from {}'.format(args.weights))
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
    model = model.cuda()
    
    logging.info("param size = %.5fMB" % (count_parameters_in_MB(model)))
    flops = calculate_FLOPs_in_M(model, (1, args.img_chs, 256, 256))
    logging.info('FLOPs: {:.4f}M'.format(flops))

    if os.path.isfile(args.list_dir):
        TEST_LIST_FILE = args.list_dir
    else:
        TEST_LIST_FILE = os.path.join(args.list_dir, 'test.txt')
        assert os.path.isfile(TEST_LIST_FILE)
    COVER_DIR = args.cover_dir
    STEGO_DIR = args.stego_dir
    assert args.batch_size % 2 == 0
    test_dataset = CustomDataset_color(TEST_LIST_FILE, COVER_DIR, STEGO_DIR)
    batch_size_loader = args.batch_size
    logging.info("test_datas: %d" % (len(test_dataset)))
    test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_loader, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    
    start = time.time()
    test_acc, test_wauc, test_md5 = validate(test_queue, model)
    logging.info('Epoch: {}, Test_acc: {:.3f}. Test_wauc: {:.5f}, Test_md5: {:.5f}'.format(epoch, test_acc, test_wauc, test_md5))
    logging.info('Test time: %ds.' % (time.time() - start))


def validate(val_queue, model):
    top1 = AverageMeter()
    model.eval()
    all_labels = np.array([])
    all_predicts = np.array([])
    
    for x, target in tqdm.tqdm(val_queue):
        shape = list(x.size())
        if len(shape) == 5:
            x = x.reshape(shape[0] * shape[1], *shape[2:])
            target = target.reshape(-1)
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        with torch.no_grad():
            logits = model(x)
        
        prec1 = accuracy_2cls(logits, target)
        test_pred = logits.softmax(dim=1)[:, 1]
        all_labels = np.concatenate([all_labels, target.cpu().detach().numpy()], axis=0)
        all_predicts = np.concatenate([all_predicts, test_pred.cpu().detach().numpy()], axis=0)
        
        n = x.size(0)
        top1.update(prec1.item(), n)
    
    wauc = alaska_weighted_auc(all_labels, all_predicts)
    eval_criter = tf_roc(all_predicts, all_labels, 9999)
    pmd_05 = eval_criter.get_pmd(0.05)
    return top1.avg, wauc, pmd_05

if __name__ == '__main__':
    main()
