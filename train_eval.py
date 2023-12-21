import os
import sys
import time
import glob
import logging
import argparse
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tools.utils import AverageMeter, accuracy_2cls
from tools.utils import count_parameters_in_MB, count_params_flops
from tools.utils import create_exp_dir, save_checkpoint
from tools.flops_benchmark import calculate_FLOPs_in_M
from models.model_eval_steg_op_depth import NetworkGC, NetworkGCCfg
from parse_s2 import get_op_and_depth_weights
from parse_s2 import parse_architecture
from dataset.stegdatas import CustomDataset_color
from tools.optims import get_optimizer, get_scheduler



parser = argparse.ArgumentParser("training the searched architecture on stegs")
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
# parser.add_argument('--data_load', type=str, default='custom', choices=['mydata', 'custom'], help='class to read data')

parser.add_argument('--model_path', type=str, default='', help='the searched model path')
parser.add_argument('--config_path', type=str, default='', help='the model config path')
# parser.add_argument('--model_name', type=str, default='', help='the model name')
parser.add_argument('--save', type=str, default='./checkpoints/', help='model and log saving path')

parser.add_argument('--resume_derived_model', type=str, default='', help='exported model path from searched model')

# training hyper-parameters
parser.add_argument('--print_freq', type=float, default=100, help='print frequency')
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--epochs', type=int, default=250, help='num of total training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate')
parser.add_argument('--start_nopc_epoch', type=int, default=150, help='start no pair constraint epoch')

'''optimizer'''
parser.add_argument('--optim', type=str, default='adamax', help='optimizer')
parser.add_argument('--init_lr', type=float, default=0.001, help='train batch_size')
parser.add_argument('--optim_params', type=str, default='dict(weight_decay=1e-4, eps=1e-8)', help='optim parameters')
'''scheduler'''
parser.add_argument('--scheduler', type=str, default='multisteplr', help='scheduler')
parser.add_argument('--scheduler_params', type=str, default='dict(milestones=[140], gamma=0.1)', help='scheduler parameters')

# others
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')


args, unparsed = parser.parse_known_args()

args.save = os.path.join(args.save, 'train-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.note))
create_exp_dir(args.save, scripts_to_save=None)

code_save = os.path.join(args.save, 'codes')
create_exp_dir(code_save, scripts_to_save=None)
os.system('cp ./*.py  ./*.sh  ' + code_save)
os.system('cp -r ./dataset/  ./models/ ./tools/  ./configs/  ' + code_save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, xs, targets):
        log_probs = self.logsoftmax(xs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    set_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    
    # create model
    logging.info('parsing the architecture')
    if args.model_path and os.path.isfile(args.model_path):
        op_weights, depth_weights = get_op_and_depth_weights(args.model_path)
        parsed_arch = parse_architecture(op_weights, depth_weights)
        model = NetworkGC(args.num_classes, args.img_chs, parsed_arch, args.dropout_rate)
    elif args.config_path and os.path.isfile(args.config_path):
        model_config = json.load(open(args.config_path, 'r'))
        model = NetworkGCCfg(args.num_classes, args.img_chs, model_config, args.dropout_rate)

    else:
        raise Exception('invalid --model_path and --config_path')
    # model = nn.DataParallel(model)
    model = model.cuda()
    if isinstance(model, nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    if args.model_path or args.config_path:
        config = model_module.config
        with open(os.path.join(args.save, 'model.config'), 'w') as f:
            json.dump(config, f, indent=4)
    # logging.info(config)
    logging.info("param size = %.5fMB", count_parameters_in_MB(model))
    flops = calculate_FLOPs_in_M(model, (1, args.img_chs, 256, 256))
    logging.info('FLOPs: {:.4f}M'.format(flops))

    # thop
    _, macs = count_params_flops(model, (1, args.img_chs, 256, 256))
    logging.info('macs: {:.4f}M'.format(macs))
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    if args.label_smooth > 0.0:
        criterion_smooth = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
        criterion_smooth = criterion_smooth.cuda()
    else:
        criterion_smooth = nn.CrossEntropyLoss()
        criterion_smooth = criterion_smooth.cuda()
    
    # define transform and initialize dataloader
    TRAIN_LIST_FILE = os.path.join(args.list_dir, 'train.txt')
    VALID_LIST_FILE = os.path.join(args.list_dir, 'valid.txt')
    COVER_DIR = args.cover_dir
    STEGO_DIR = args.stego_dir
    assert args.batch_size % 2 == 0
    train_dataset = CustomDataset_color(TRAIN_LIST_FILE, COVER_DIR, STEGO_DIR, train_flag=True)
    valid_dataset = CustomDataset_color(VALID_LIST_FILE, COVER_DIR, STEGO_DIR)
    batch_size_loader = args.batch_size
    logging.info("train_datas: %d" % (len(train_dataset)))
    logging.info("valid_datas: %d" % (len(valid_dataset)))
    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_loader, shuffle=False,
                                              pin_memory=True, num_workers=args.workers)
    val_queue = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size_loader, shuffle=False,
                                            pin_memory=True, num_workers=args.workers)

    batches_in_epoch = len(train_queue)

    net_params = model.parameters()
    optimizer = get_optimizer(net_params, optim_name=args.optim, init_lr=args.init_lr, optim_params=args.optim_params)
    scheduler = get_scheduler(optimizer, scheduler_name=args.scheduler, scheduler_params_str=args.scheduler_params,
    						  args=args, batches_in_epoch=batches_in_epoch)

    # define learning rate scheduler
    best_acc = 0.0
    best_epoch = 0
    start_epoch = 0
    
    if args.start_nopc_epoch is None:
        args.start_nopc_epoch = args.epochs
    logging.info("Prepare to training %d epochs PC + %d epochs noPC !"
    		%(args.start_nopc_epoch, args.epochs-args.start_nopc_epoch))

    if args.resume_derived_model:
        # assert os.path.isfile(args.resume_derived_model)
        logging.info('loading exported model from {}'.format(args.resume_derived_model))
        checkpoint = torch.load(args.resume_derived_model)
        model.load_state_dict(checkpoint['state_dict'])
    
    # the main loop
    for epoch in range(start_epoch, args.epochs):
    
        if args.start_nopc_epoch <= epoch:
            if args.start_nopc_epoch == epoch:
                # train_dataset.shuffle_pair(nopc=True)
                train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_loader, shuffle=True,
                										  pin_memory=True, num_workers=args.workers)
            if args.start_nopc_epoch == epoch:
                logging.info("Start no pc !")

        
        current_lr = scheduler.get_lr()[0]
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
        
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        
        val_acc, val_obj = validate(val_queue, model, criterion)
        
        logging.info('Epoch: %d, Train_loss: %.4f, Train_acc: %.3f, Val_loss: %.4f, Val_acc: %.3f, lr %e, Epoch time: %ds.' %
                 (epoch, train_obj, train_acc, val_obj, val_acc, current_lr, time.time() - epoch_start))
        
        is_best = False
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)
        
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        scheduler.step()
    logging.info("Best epoch: %d, val_acc: %.3f"%(best_epoch, best_acc))


def train(train_queue, model, criterion, optimizer):
    objs = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    model.train()
    
    end = time.time()
    iters = len(train_queue)
    for step, (x, target) in enumerate(train_queue):
    
        shape = list(x.size())
        if len(shape) == 5:
            x = x.reshape(shape[0] * shape[1], *shape[2:])
            target = target.reshape(-1)
        
        data_time.update(time.time() - end)
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # forward
        batch_start = time.time()
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, target)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - batch_start)
        
        prec1 = accuracy_2cls(logits, target)
        n = x.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        
        if step % args.print_freq == 0:
            duration = 0 if step == 0 else time.time() - duration_start
            duration_start = time.time()
            # logging.info('TRAIN Step: %03d loss: %e acc: %f Duration: %ds BTime: %.3fs DTime: %.4fs',
            # 						step, objs.avg, top1.avg, duration, batch_time.avg, data_time.avg)
            print('TRAIN Step: %03d/%03d loss: %.5f acc: %.4f Duration: %ds BTime: %.3fs DTime: %.4fs' %
                  (step, iters, objs.avg, top1.avg, duration, batch_time.avg, data_time.avg))
        end = time.time()
    
    return top1.avg, objs.avg


def validate(val_queue, model, criterion):
    objs = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    iters = len(val_queue)
    for step, (x, target) in enumerate(val_queue):
    
        shape = list(x.size())
        if len(shape) == 5:
            x = x.reshape(shape[0] * shape[1], *shape[2:])
            target = target.reshape(-1)
        
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, target)
        
        prec1 = accuracy_2cls(logits, target)
        n = x.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        
        if step % args.print_freq == 0:
            duration = 0 if step == 0 else time.time() - duration_start
            duration_start = time.time()
            # logging.info('VALID Step: %03d loss: %e acc: %f Duration: %ds', step, objs.avg, top1.avg, duration)
            print('VALID Step: %03d/%03d loss: %.5f acc: %.4f Duration: %ds' % (step, iters, objs.avg, top1.avg, duration))
    
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
