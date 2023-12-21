import os
import sys
import time
import glob
import logging
import argparse
import pickle
import copy
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tools.utils import AverageMeter, accuracy_2cls
from tools.utils import count_parameters_in_MB
from tools.utils import create_exp_dir
from models.model_search_steg_op_depth import NAS_GroupConv
from dataset.stegdatas import CustomDataset_color
from tools.optims import get_optimizer, get_scheduler

parser = argparse.ArgumentParser("searching TF-NAS")
# various path
parser.add_argument('--cover_dir', type=str, default='../cover_data', help='location of the cover_data corpus')
parser.add_argument('--stego_dir', type=str, default='../stego_data', help='location of the stego_data corpus')
parser.add_argument('--list_dir', type=str, default='.', help='data list folder like bb/train.txt')
parser.add_argument('--img_chs', type=int, default=3, help='input channels number')
parser.add_argument('--num_classes', type=int, default=2, help='class number of training set')
# parser.add_argument('--normalize', type=int, default=0, choices=[0, 1], help='Normalize images')
# parser.add_argument('--aug', type=int, default=1, choices=[0, 1], help='augment dataset: flip and rot')
# parser.add_argument('--data_load', type=str, default='custom', choices=['custom'], help='class to read data')

parser.add_argument('--save', type=str, default='./checkpoints', help='model and log saving path')

# training hyper-parameters
parser.add_argument('--print_freq', type=float, default=100, help='print frequency / iter')
parser.add_argument('--valid_freq', type=int, default=1, help='valid frequency / epoch')
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--epochs', type=int, default=120, help='num of total training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--a_lr', type=float, default=0.01, help='learning rate for arch')
parser.add_argument('--a_wd', type=float, default=5e-4, help='weight decay for arch')
parser.add_argument('--a_beta1', type=float, default=0.5, help='beta1 for arch')
parser.add_argument('--a_beta2', type=float, default=0.999, help='beta2 for arch')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--T', type=float, default=5.0, help='temperature for gumbel softmax')
parser.add_argument('--T_decay', type=float, default=0.96, help='temperature decay')

parser.add_argument('--wm_epoch', type=int, default=20, help='warmup epochs')
parser.add_argument('--wm_sample', type=str, default='all', choices=['gumbel', 'all'], help='warmup sampling mode')

parser.add_argument('--resume', type=str, default=None, help='resume model for finetuning')
# parser.add_argument('--model_path', type=str, default='', help='the searched model path from stage1()')
# parser.add_argument('--config_path', type=str, default='', help='the model config path from stage1()')

'''optimizer'''
parser.add_argument('--optim_w', type=str, default='adamax', help='optimizer')
parser.add_argument('--w_lr', type=float, default=0.001, help='learning rate for weights')
parser.add_argument('--optim_w_params', type=str, default="dict(weight_decay=1e-4, eps=1e-8)", help='optim_w arguments')
'''scheduler'''
parser.add_argument('--scheduler', type=str, default='multisteplr', help='scheduler')
parser.add_argument('--scheduler_params', type=str, default="dict(milestones=[100], gamma=0.1)", help='scheduler parameters')
# others
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--start_nopc_epoch', type=int, default=110, help='start no pair constraint epoch')

args, unparsed = parser.parse_known_args()

args.save = os.path.join(args.save, 's2-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.note))
create_exp_dir(args.save, scripts_to_save=None)

code_save = os.path.join(args.save, 'codes')
create_exp_dir(code_save, scripts_to_save=None)
os.system('cp ./*.py  ./*.sh  ' + code_save)
os.system('cp -r ./dataset/ ./models/ ./tools/  ./configs/ ' + code_save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled=True
    cudnn.benchmark = True
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    # if args.config_path and os.path.isfile(args.config_path):
    #     model_config = json.load(open(args.config_path, 'r'))
    model = NAS_GroupConv(args.num_classes, img_chs=args.img_chs, p=0.0)

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        if 'epoch' in ckpt.keys():
            start_epoch = ckpt['epoch'] + 1
        else:
            start_epoch = 0
        logging.info("resume %s"%(args.resume))
    else:
        start_epoch = 0
    model = model.cuda()
    if isinstance(model, nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    logging.info("param size = %fMB", count_parameters_in_MB(model))
    
    # save initial model
    model_path = os.path.join(args.save, 'searched_model_00.pth.tar')
    torch.save({
        'state_dict': model.state_dict(),
    }, model_path)
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    net_w_params = model_module.weight_parameters()

    optimizer_w = get_optimizer(net_w_params, optim_name=args.optim_w, init_lr=args.w_lr, optim_params=args.optim_w_params)
    assert args.scheduler == "multisteplr"
    scheduler_params = eval(args.scheduler_params)
    milestones = scheduler_params.get("milestones")
    assert isinstance(milestones, list)

    TRAIN_LIST_FILE = os.path.join(args.list_dir, 'search_w.txt')
    VALID_LIST_FILE = os.path.join(args.list_dir, 'search_a.txt')
    # TRAIN_LIST_FILE = os.path.join(args.list_dir, 'train.txt')
    # VALID_LIST_FILE = os.path.join(args.list_dir, 'valid.txt')
    COVER_DIR = args.cover_dir
    STEGO_DIR = args.stego_dir
    assert args.batch_size % 2 == 0
    train_dataset = CustomDataset_color(TRAIN_LIST_FILE, COVER_DIR, STEGO_DIR, train_flag=True)
    valid_dataset = CustomDataset_color(VALID_LIST_FILE, COVER_DIR, STEGO_DIR)
    batch_size_loader = args.batch_size
    logging.info("train_datas: %d" % (len(train_dataset)))
    logging.info("train_arch_datas: %d" % (len(valid_dataset)))
    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_loader, shuffle=False,
                                              pin_memory=True, num_workers=args.workers)
    val_queue = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size_loader, shuffle=False,
                                            pin_memory=True, num_workers=args.workers)
    
    best_acc = 0.0
    best_epoch = 0
    best_arch_acc = 0.0
    best_arch_epoch = 0
    
    if args.start_nopc_epoch is None:
        args.start_nopc_epoch = args.epochs
    logging.info("Prepare to training %d epochs PC + %d epochs noPC !"
        %(args.start_nopc_epoch, args.epochs-args.start_nopc_epoch))
    
    for epoch in range(start_epoch, args.epochs):
    
        if args.start_nopc_epoch <= epoch:
            if args.start_nopc_epoch == epoch:
                # train_dataset.shuffle_pair(nopc=True)
                train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_loader, shuffle=True,
                										  pin_memory=True, num_workers=args.workers)
            if args.start_nopc_epoch == epoch:
                logging.info("Start no pc !")
        
        model_module.set_temperature(args.T)
        # adjust lr
        if epoch in milestones:
            for param_group in optimizer_w.param_groups:
                param_group['lr'] /= 10
        # lr = scheduler.get_lr()[0]
        lr = optimizer_w.param_groups[0]['lr']
        if model_module.arch_parameters() != []:
            optimizer_a = torch.optim.Adam(
                model_module.arch_parameters(),
                lr = args.a_lr,
                betas = (args.a_beta1, args.a_beta2),
                weight_decay = args.a_wd)
        
        # training
        epoch_start = time.time()
        if epoch < args.wm_epoch:
            loss_w, train_acc = train_wo_arch(train_queue, model, criterion, optimizer_w, args.wm_sample)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch: %d, loss_W: %.5f, Train_acc: %.3f, lr: %e,  Epoch time: %ds',
                    epoch, loss_w, train_acc, lr, epoch_duration)
        else:
            if epoch == args.wm_epoch:
                logging.info("Start Arch updating!")
            loss_w, loss_a, train_acc, train_arch_acc = \
                train_w_arch(train_queue, val_queue, model, criterion, optimizer_w, optimizer_a)
            args.T *= args.T_decay
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch: %d, train_step: loss_W: %.5f, loss_A: %.5f, acc_W: %.3f, acc_A: %.3f, lr: %e, T: %e, Epoch time: %ds',
                    epoch, loss_w, loss_a, train_acc, train_arch_acc, lr, args.T, epoch_duration)
            
            if train_arch_acc > best_arch_acc:
                best_arch_acc = train_arch_acc
                best_arch_epoch = epoch
                model_path = os.path.join(args.save, 'searched_model_arch_best.pth.tar')
                torch.save({
                    'state_dict': model.state_dict(),
                    'alpha': model_module.arch_parameters(),
                    # 'optimizer_w': optimizer_w.state_dict(),
                    # 'optimizer_a': optimizer_a.state_dict(),
                    'epoch': best_arch_epoch,
                }, model_path)
            
            # logging arch parameters
            logging.info('The current arch parameters are:')
            for param in model_module.log_alphas_parameters():
                param = np.exp(param.detach().cpu().numpy())
                logging.info(' '.join(['{:.6f}'.format(p) for p in param]))
            for param in model_module.betas_parameters():
                param = F.softmax(param.detach().cpu(), dim=-1)
                param = param.numpy()
                logging.info(' '.join(['{:.6f}'.format(p) for p in param]))

        if args.epochs - epoch < 5 or (epoch+1)%args.valid_freq==0:
            val_acc = validate(val_queue, model, criterion, _eval=False)
            logging.info('Epoch: %d, Val_acc: %.3f', epoch, val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                model_path = os.path.join(args.save, 'searched_model_best.pth.tar')
                torch.save({
                    'state_dict': model.state_dict(),
                    'alpha': model_module.arch_parameters(),
                    # 'optimizer_w': optimizer_w.state_dict(),
                    'epoch': best_epoch,
                }, model_path)

        # save model
        model_path = os.path.join(args.save, 'searched_model_last.pth.tar')
        torch.save({
            'state_dict': model.state_dict(),
            'alpha': model_module.arch_parameters(),
            'epoch': epoch,
        }, model_path)
        
        # scheduler.step()
    logging.info('Best epoch: %d, Valid_acc %.3f', best_epoch, best_acc)


def train_wo_arch(train_queue, model, criterion, optimizer_w, wm_sample='gumbel'):
    objs_w = AverageMeter()
    top1   = AverageMeter()
    
    if isinstance(model, nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    model.train()
    
    for param in model_module.weight_parameters():
        param.requires_grad = True
    for param in model_module.arch_parameters():
        param.requires_grad = False
    
    for step, (x_w, target_w) in enumerate(train_queue):
    
        shape = list(x_w.size())
        if len(shape) == 5:
            x_w = x_w.reshape(shape[0] * shape[1], *shape[2:])
            target_w = target_w.reshape(-1)
        x_w = x_w.cuda(non_blocking=True)
        target_w = target_w.cuda(non_blocking=True)
        
        if wm_sample == 'gumbel':
            logits_w = model(x_w, sampling=True, mode='gumbel')
            if isinstance(logits_w, tuple):
                logits_w = logits_w[0]
            # reset switches of log_alphas
            model_module.reset_switches()
        elif wm_sample == 'all':
            logits_w = model(x_w, sampling=False)
            if isinstance(logits_w, tuple):
                logits_w = logits_w[0]
        
        loss_w = criterion(logits_w, target_w)
        
        optimizer_w.zero_grad()
        loss_w.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model_module.weight_parameters(), args.grad_clip)
        optimizer_w.step()
        
        prec1 = accuracy_2cls(logits_w, target_w)
        n = x_w.size(0)
        objs_w.update(loss_w.item(), n)
        top1.update(prec1.item(), n)
        
        if step % args.print_freq == 0:
            # logging.info('Train wo_Arch Step: %04d loss: %f acc: %f', step, objs_w.avg, top1.avg)
            print('Train wo_Arch Step: %04d loss: %.4f acc: %.3f' % (step, objs_w.avg, top1.avg))
    
    return objs_w.avg, top1.avg


def train_w_arch(train_queue, val_queue, model, criterion, optimizer_w, optimizer_a):
    objs_a = AverageMeter()
    objs_w = AverageMeter()
    top1_w = AverageMeter()
    top1_a = AverageMeter()
    
    if isinstance(model, nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    model.train()
    
    for step, (x_w, target_w) in enumerate(train_queue):
        shape = list(x_w.size())
        if len(shape) == 5:
            x_w = x_w.reshape(shape[0] * shape[1], *shape[2:])
            target_w = target_w.reshape(-1)
        x_w = x_w.cuda(non_blocking=True)
        target_w = target_w.cuda(non_blocking=True)
        
        for param in model_module.weight_parameters():
            param.requires_grad = True
        for param in model_module.arch_parameters():
            param.requires_grad = False
        
        logits_w_gumbel = model(x_w, sampling=True, mode='gumbel')
        if isinstance(logits_w_gumbel, tuple):
            logits_w_gumbel = logits_w_gumbel[0]
        loss_w_gumbel = criterion(logits_w_gumbel, target_w)
        logits_w_random = model(x_w, sampling=True, mode='random')
        if isinstance(logits_w_random, tuple):
            logits_w_random = logits_w_random[0]
        loss_w_random = criterion(logits_w_random, target_w)
        loss_w = loss_w_gumbel + loss_w_random
        
        optimizer_w.zero_grad()
        loss_w.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model_module.weight_parameters(), args.grad_clip)
        optimizer_w.step()
        
        prec1 = accuracy_2cls(logits_w_gumbel, target_w)
        n = x_w.size(0)
        objs_w.update(loss_w.item(), n)
        top1_w.update(prec1.item(), n)
        
        if step % 2 == 0:
            # optimize a
            try:
                x_a, target_a = next(val_queue_iter)
            except:
                val_queue_iter = iter(val_queue)
                x_a, target_a = next(val_queue_iter)
            shape = list(x_a.size())
            if len(shape) == 5:
                x_a = x_a.reshape(shape[0] * shape[1], *shape[2:])
                target_a = target_a.reshape(-1)
            x_a = x_a.cuda(non_blocking=True)
            target_a = target_a.cuda(non_blocking=True)
            
            for param in model_module.weight_parameters():
                param.requires_grad = False
            for param in model_module.arch_parameters():
                param.requires_grad = True
            
            logits_a = model(x_a, sampling=False)
            if isinstance(logits_a, tuple):
                logits_a = logits_a[0]
            loss_a = criterion(logits_a, target_a)	# update_arch阶段，更新所有的arch参数
            loss = loss_a
            
            optimizer_a.zero_grad()
            loss.backward()
            # loss_a.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model_module.arch_parameters(), args.grad_clip)
            optimizer_a.step()
            
            prec1 = accuracy_2cls(logits_a, target_a)
            
            # ensure log_alphas to be a log probability distribution
            for log_alphas in model_module.arch_parameters():
                log_alphas.data = F.log_softmax(log_alphas.detach().data, dim=-1)
            
            n = x_a.size(0)
            objs_a.update(loss_a.item(), n)
            top1_a.update(prec1.item(), n)
        
        if step % args.print_freq == 0:
            # logging.info('Train w_Arch Step: %04d loss_W: %f acc: %f loss_A: %f',
            # 			  step, objs_w.avg, top1_w.avg, objs_a.avg)
            print('Train w_Arch Step: %04d loss_W: %f acc_W: %f loss_A: %f acc_W: %f' %
                  (step, objs_w.avg, top1_w.avg, objs_a.avg, top1_a.avg))
    
    return objs_w.avg, objs_a.avg, top1_w.avg, top1_a.avg


def validate(val_queue, model, criterion, _eval=False):
    objs = AverageMeter()
    top1 = AverageMeter()
    if isinstance(model, nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    # model.eval()
    # disable moving average
    model.eval() if _eval else model.train() # disable running stats for projection, 此处应该使用model.train()状态，避免使用running_mean和running_var来做归一化
    
    for step, (x, target) in enumerate(val_queue):
        shape = list(x.size())
        if len(shape) == 5:
            x = x.reshape(shape[0] * shape[1], *shape[2:])
            target = target.reshape(-1)
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(x, sampling=True, mode='gumbel')
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, target)
        # reset switches of log_alphas
        model_module.reset_switches()
        
        prec1 = accuracy_2cls(logits, target)
        n = x.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        
        if step % args.print_freq == 0:
            # logging.info('Valid Step: %04d loss: %f acc: %f', step, objs.avg, top1.avg)
            print('Valid Step: %04d loss: %f acc: %f' % (step, objs.avg, top1.avg))
    
    return top1.avg


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
