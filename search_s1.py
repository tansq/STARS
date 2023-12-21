
import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tools.utils import create_exp_dir, AverageMeter, accuracy_2cls, count_parameters_in_MB
from models.model_distill import SRMSep_stem
from models.model_search_steg_op_depth import NAS_GroupConv_stage1
from dataset.stegdatas import CustomDataset_color
from tools.optims import get_optimizer, get_scheduler

parser = argparse.ArgumentParser("distill in the first stage")

parser.add_argument('--cover_dir', type=str, default='../cover_data', help='location of the cover_data corpus')
parser.add_argument('--stego_dir', type=str, default='../stego_data', help='location of the stego_data corpus')
parser.add_argument('--list_dir', type=str, default='.', help='data list folder like bb/train.txt')
parser.add_argument('--img_chs', type=int, default=3, help='input channels number')
parser.add_argument('--num_classes', type=int, default=2, help='class number of training set')
# parser.add_argument('--normalize', type=int, default=0, choices=[0, 1], help='Normalize images')
# parser.add_argument('--aug', type=int, default=1, choices=[0, 1], help='augment dataset: flip and rot')
# parser.add_argument('--data_load', type=str, default='custom', choices=['custom'], help='class to read data')

parser.add_argument('--save', type=str, default='./checkpoints/', help='model and log saving path')
parser.add_argument('--print_freq', type=float, default=100, help='print frequency')
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--epochs', type=int, default=120, help='num of total training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--resume', type=str, default=None, help='resume model for finetuning')

parser.add_argument('--teacher_resume', type=str, default=None, help='resume teacher weights')

'''optimizer'''
parser.add_argument('--optim_w', type=str, default='adamax', help='optimizer')
parser.add_argument('--w_lr', type=float, default=0.001, help='train batch_size')
parser.add_argument('--optim_w_params', type=str, default='dict(weight_decay=1e-4, eps=1e-8)', help='optim parameters')

parser.add_argument('--a_lr', type=float, default=0.01, help='learning rate for arch')
parser.add_argument('--a_wd', type=float, default=5e-4, help='weight decay for arch')
parser.add_argument('--a_beta1', type=float, default=0.5, help='beta1 for arch')
parser.add_argument('--a_beta2', type=float, default=0.999, help='beta2 for arch')

'''scheduler'''
parser.add_argument('--scheduler', type=str, default='multisteplr', help='scheduler')
parser.add_argument('--scheduler_params', type=str, default='dict(milestones=[100], gamma=0.1)', help='scheduler parameters')

parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--T', type=float, default=5.0, help='temperature for gumbel softmax')
parser.add_argument('--T_decay', type=float, default=0.96, help='temperature decay')

parser.add_argument('--wm_epoch', type=int, default=0, help='warmup epochs')
parser.add_argument('--wm_sample', type=str, default='gumbel', choices=['gumbel', 'all'], help='warmup sampling mode')

# others
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--start_nopc_epoch', type=int, default=90, help='start no pair constraint epoch')

args, unparsed = parser.parse_known_args()

args.save = os.path.join(args.save, 's1-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.note))
create_exp_dir(args.save, scripts_to_save=None)

code_save = os.path.join(args.save, 'codes')
create_exp_dir(code_save, scripts_to_save=None)
os.system('cp ./*.py  ./*.sh  ' + code_save)
os.system('cp -r ./dataset/ ./models/ ./tools/  ./configs/  ' + code_save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = SRMSep_stem(img_chs=3, affine=False)
    if args.teacher_resume is not None:
        ckpt = torch.load(args.teacher_resume, map_location='cpu')
        teacher.load_state_dict(ckpt['state_dict'])
    teacher = teacher.cuda()
    model = NAS_GroupConv_stage1(args.num_classes, args.img_chs)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        if 'epoch' in ckpt.keys():
            start_epoch = ckpt['epoch'] + 1
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    model.to(device)
    # model = model.cuda()

    logging.info("param size = %fMB", count_parameters_in_MB(model))

    # save initial model
    model_path = os.path.join(args.save, 'searched_model_00.pth.tar')
    torch.save({
        'state_dict': model.state_dict(),
    }, model_path)

    criterion = nn.CrossEntropyLoss().cuda()

    # get lr list
    if "NAS" in model.__class__.__name__:
        net_params = model.weight_parameters()
    else:
        net_params = model.parameters()
    optimizer_w = get_optimizer(net_params, optim_name=args.optim_w, init_lr=args.w_lr, optim_params=args.optim_w_params)
    assert args.scheduler == "multisteplr"
    scheduler_params = eval(args.scheduler_params)
    milestones = scheduler_params.get("milestones")
    assert isinstance(milestones, list)

    if callable(getattr(model, "arch_parameters", None)):
        print("init optimizer_a")
        if model.arch_parameters() != []:
            optimizer_a = torch.optim.Adam(model.arch_parameters(), lr=args.a_lr,
                                           betas=(args.a_beta1, args.a_beta2), weight_decay=args.a_wd)

    # define transform and initialize dataloader
    if 'NAS' in model.__class__.__name__:
        TRAIN_LIST_FILE = os.path.join(args.list_dir, 'search_w.txt')
        VALID_LIST_FILE = os.path.join(args.list_dir, 'search_a.txt')
    else:
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

    if args.start_nopc_epoch is None:
        args.start_nopc_epoch = args.epochs
    logging.info("Prepare to training %d epochs PC + %d epochs noPC !"
                 % (args.start_nopc_epoch, args.epochs - args.start_nopc_epoch))

    if callable(getattr(model, "weight_parameters", None)):
        eval_mode = False
    else:
        eval_mode = True
    best_acc = 0.0
    best_epoch = 0
    best_arch_acc = 0.0
    best_arch_epoch = 0
    print("Start Training")
    for epoch in range(start_epoch, args.epochs):
        if callable(getattr(model, "set_temperature", None)):
            model.set_temperature(args.T)   # for NAS

        if args.start_nopc_epoch <= epoch:
            if args.start_nopc_epoch == epoch:     # custom dataset
                train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_loader, shuffle=True,
                                                          pin_memory=True, num_workers=args.workers)
            if args.start_nopc_epoch == epoch:
                logging.info("Start no pc !")

        # adjust lr
        if epoch in milestones:
            for param_group in optimizer_w.param_groups:
                param_group['lr'] /= 10

        # lr = scheduler.get_lr()[0]
        lr = optimizer_w.param_groups[0]['lr']

        # search
        if callable(getattr(model, "weight_parameters", None)):
            if epoch < args.wm_epoch:
                loss_w, train_acc = \
                    train_wo_arch(train_queue, model, teacher, criterion, optimizer_w, device, wm_sample=args.wm_sample)
                logging.info('Epoch: %d, loss_W: %.4f, Train_acc: %.3f, lr: %e',
                             epoch, loss_w, train_acc, lr)
            else:
                if epoch == args.wm_epoch:
                    logging.info("Start Arch updating!")
                loss_w, loss_a, train_acc, train_arch_acc = \
                    train_w_arch(train_queue, val_queue, model, teacher, criterion, optimizer_w, optimizer_a, device)
                args.T *= args.T_decay
                logging.info(
                    'Epoch: %d, train_step: loss_W: %.4f, loss_A: %.4f, acc_W: %.3f, acc_A: %.3f, lr: %e, T: %e',
                    epoch, loss_w, loss_a, train_acc, train_arch_acc, lr, args.T)

                # save best arch with best_arch_acc
                if train_arch_acc > best_arch_acc:
                    best_arch_acc = train_arch_acc
                    best_arch_epoch = epoch
                    model_path = os.path.join(args.save, 'searched_model_arch_best.pth.tar')
                    torch.save({
                        'state_dict': model.state_dict(),
                        'alpha': model.arch_parameters(),
                        # 'optimizer_w': optimizer_w.state_dict(),
                        # 'optimizer_a': optimizer_a.state_dict(),
                        'epoch': best_arch_epoch,
                    }, model_path)

                # logging arch parameters
                logging.info('The current arch parameters are:')
                log_alpha_list = []
                for param in model.log_alphas_parameters():
                    param = np.exp(param.detach().cpu().numpy())
                    log_alpha_list.append(param)
                    logging.info(' '.join(['{:.6f}'.format(p) for p in param]))
                if callable(getattr(model, "betas_parameters", None)):
                    for param in model.betas_parameters():
                        param = F.softmax(param.detach().cpu(), dim=-1)
                        param = param.numpy()
                        logging.info(' '.join(['{:.6f}'.format(p) for p in param]))
        else:
            train_obj, train_acc = train(train_queue, model, teacher, criterion, optimizer_w, device)
            # logging.info('Epoch: %d, Train_loss: %.4f, Train_acc %.3f, lr: %e',
            #              epoch, train_obj, train_acc, lr)

        val_obj, val_acc = validate(val_queue, model, teacher, criterion, device, _eval=eval_mode)
        if callable(getattr(model, "arch_parameters", None)):
            cur_state_dict = {
                        'state_dict': model.state_dict(),
                        'alpha': model.arch_parameters(),
                        # 'optimizer_w': optimizer_w.state_dict(),
                        # 'optimizer_a': optimizer_a.state_dict(),
                        'epoch': epoch,
                        }
        else:
            cur_state_dict = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
            }
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model_path = os.path.join(args.save, 'searched_model_best.pth.tar')
            torch.save(cur_state_dict, model_path)
        if callable(getattr(model, "arch_parameters", None)):
            logging.info('Epoch: %d, Val_loss: %.4f, Val_acc: %.3f', epoch, val_obj, val_acc)
        else:
            logging.info(
                'Epoch: %d, Train_loss: %.4f, Train_acc: %.3f Val_loss: %.4f, Val_acc: %.3f.' %
                (epoch, train_obj, train_acc, val_obj, val_acc))

        model_path = os.path.join(args.save, 'searched_model_last.pth.tar')
        torch.save(cur_state_dict, model_path)
    logging.info('Best epoch: %d, Valid_acc: %.3f', best_epoch, best_acc)


def train(train_queue, model, teacher, criterion, optimizer, device):
    objs = AverageMeter()
    top1 = AverageMeter()
    model.train()

    iters = len(train_queue)
    for step, (x, target) in enumerate(train_queue):

        shape = list(x.size())
        if len(shape) == 5:
            x = x.reshape(shape[0] * shape[1], *shape[2:])
            target = target.reshape(-1)

        # x = x.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        x, target = x.to(device), target.to(device)
        logits, student_feature = model(x)

        #   get teacher results
        with torch.no_grad():
            teacher_feature = teacher(x)
        if isinstance(teacher_feature, tuple):
            teacher_feature = teacher_feature[1]

        loss = torch.dist(student_feature, teacher_feature, p=2) * args.alpha     # L2 Loss
        loss += criterion(logits, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = accuracy_2cls(logits, target)
        n = x.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.print_freq == 0:
            duration = 0 if step == 0 else time.time() - duration_start
            duration_start = time.time()
            # logging.info('TRAIN Step: %03d loss: %e acc: %f Duration: %ds BTime: %.3fs DTime: %.4fs',
            # 						step, objs.avg, top1.avg, duration, batch_time.avg, data_time.avg)
            print('TRAIN Step: %03d/%03d loss: %.5f acc: %.3f Duration: %ds' %
                  (step, iters, objs.avg, top1.avg, duration))

    return objs.avg, top1.avg


def train_wo_arch(train_queue, model, teacher, criterion, optimizer, device, wm_sample='gumbel'):
    objs = AverageMeter()
    top1 = AverageMeter()
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    model.train()

    for param in model.weight_parameters():
        param.requires_grad = True
    for param in model.arch_parameters():
        param.requires_grad = False

    # end = time.time()
    iters = len(train_queue)
    for step, (x, target) in enumerate(train_queue):

        shape = list(x.size())
        if len(shape) == 5:
            x = x.reshape(shape[0] * shape[1], *shape[2:])
            target = target.reshape(-1)

        # data_time.update(time.time() - end)
        # x = x.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        x, target = x.to(device), target.to(device)

        if wm_sample == 'gumbel':
            logits, student_feature = model(x, sampling=True, mode='gumbel')
            # reset switches of log_alphas
            model.reset_switches()
        elif wm_sample == 'all':
            logits, student_feature = model(x, sampling=False)

        #   get teacher results
        with torch.no_grad():
            teacher_feature = teacher(x)
        if isinstance(teacher_feature, tuple):
            teacher_feature = teacher_feature[1]

        loss = torch.dist(student_feature, teacher_feature, p=2) * args.alpha     # L2 Loss
        loss += criterion(logits, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = accuracy_2cls(logits, target)
        n = x.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.print_freq == 0:
            duration = 0 if step == 0 else time.time() - duration_start
            duration_start = time.time()
            # logging.info('TRAIN Step: %03d loss: %e acc: %f Duration: %ds BTime: %.3fs DTime: %.4fs',
            # 						step, objs.avg, top1.avg, duration, batch_time.avg, data_time.avg)
            print('TRAIN Step: %03d/%03d loss: %.5f acc: %.3f Duration: %ds' %
                  (step, iters, objs.avg, top1.avg, duration))

    return objs.avg, top1.avg


def train_w_arch(train_queue, val_queue, model, teacher, criterion, optimizer_w, optimizer_a, device):

    objs_a = AverageMeter()
    objs_w = AverageMeter()
    top1_w = AverageMeter()
    top1_a = AverageMeter()

    model.train()

    iters = len(train_queue)
    for step, (x_w, target_w) in enumerate(train_queue):

        shape = list(x_w.size())
        if len(shape) == 5:
            x_w = x_w.reshape(shape[0] * shape[1], *shape[2:])
            target_w = target_w.reshape(-1)

        x_w = x_w.cuda(non_blocking=True)
        target_w = target_w.cuda(non_blocking=True)
        # x_w, target_w = x_w.to(device), target_w.to(device)
        for param in model.weight_parameters():
            param.requires_grad = True
        for param in model.arch_parameters():
            param.requires_grad = False

        logits_w_gumbel, student_feature_gumbel = model(x_w, sampling=True, mode='gumbel')
        loss_w_gumbel = criterion(logits_w_gumbel, target_w)
        logits_w_random, student_feature_random = model(x_w, sampling=True, mode='random')
        loss_w_random = criterion(logits_w_random, target_w)
        # loss_w = loss_w_gumbel + loss_w_random  # update_weight阶段，更新被选中路径上的权重

        #   get teacher results
        with torch.no_grad():
            teacher_feature_w = teacher(x_w)
        if isinstance(teacher_feature_w, tuple):
            teacher_feature_w = teacher_feature_w[1]

        loss_d1 = torch.dist(student_feature_gumbel, teacher_feature_w, p=2) * args.alpha     # L2 Loss
        loss_d2 = torch.dist(student_feature_random, teacher_feature_w, p=2) * args.alpha  # L2 Loss
        loss_w = loss_w_gumbel + loss_w_random + loss_d1 + loss_d2

        # backward
        optimizer_w.zero_grad()
        loss_w.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
            # x_a, target_a = x_a.to(device), target_a.to(device)

            for param in model.weight_parameters():
                param.requires_grad = False
            for param in model.arch_parameters():
                param.requires_grad = True

            logits_a, student_feature_a_gumbel = model(x_a, sampling=False)
            loss_a = criterion(logits_a, target_a)  # update_arch阶段，更新所有的arch参数

            #   get teacher results
            with torch.no_grad():
                teacher_feature_a = teacher(x_a)
            if isinstance(teacher_feature_a, tuple):
                teacher_feature_a = teacher_feature_a[1]
            loss_a += torch.dist(student_feature_a_gumbel, teacher_feature_a, p=2) * args.alpha  # L2 Loss

            optimizer_a.zero_grad()
            loss_a.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)
            optimizer_a.step()

            prec1 = accuracy_2cls(logits_a, target_a)

            # ensure log_alphas to be a log probability distribution
            for log_alphas in model.arch_parameters():
                log_alphas.data = F.log_softmax(log_alphas.detach().data, dim=-1)

            n = x_a.size(0)
            objs_a.update(loss_a.item(), n)
            top1_a.update(prec1.item(), n)

        if step % args.print_freq == 0:
            # logging.info('Train w_Arch Step: %04d loss_W: %f acc: %f loss_A: %f',
            # 			  step, objs_w.avg, top1_w.avg, objs_a.avg)
            duration = 0 if step == 0 else time.time() - duration_start
            duration_start = time.time()
            print('Train w_Arch Step: %03d/%03d loss_W: %.4f acc_W: %.3f loss_A: %.4f acc_W: %.3f Duration: %ds' %
                  (step, iters, objs_w.avg, top1_w.avg, objs_a.avg, top1_a.avg, duration))

    return objs_w.avg, objs_a.avg, top1_w.avg, top1_a.avg


def validate(val_queue, model, teacher, criterion, device, _eval=False):
    objs = AverageMeter()
    top1 = AverageMeter()
    # disable moving average
    model.eval() if _eval else model.train()  # disable running stats for projection, 此处应该使用model.train()状态，避免使用running_mean和running_var来做归一化
    teacher.eval()
    iters = len(val_queue)
    for step, (x, target) in enumerate(val_queue):

        shape = list(x.size())
        if len(shape) == 5:
            x = x.reshape(shape[0] * shape[1], *shape[2:])
            target = target.reshape(-1)

        # x = x.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        x, target = x.to(device), target.to(device)

        with torch.no_grad():

            if callable(getattr(model, "weight_parameters", None)):
                logits, student_feature = model(x, sampling=True, mode='gumbel')    # for NAS
            else:
                logits, student_feature = model(x)    # for single model
            teacher_feature = teacher(x)
            if isinstance(teacher_feature, tuple):
                teacher_feature = teacher_feature[1]

            loss = torch.dist(student_feature, teacher_feature, p=2) * args.alpha  # L2 Loss
            loss += criterion(logits, target)

        # reset switches of stem_log_alphas
        if callable(getattr(model, "weight_parameters", None)):
            model.reset_switches()

        prec1 = accuracy_2cls(logits, target)
        n = x.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.print_freq == 0:
            duration = 0 if step == 0 else time.time() - duration_start
            duration_start = time.time()
            # logging.info('VALID Step: %03d loss: %e acc: %f Duration: %ds', step, objs.avg, top1.avg, duration)
            print('VALID Step: %03d/%03d loss: %.4f acc: %.3f Duration: %ds' % (
            step, iters, objs.avg, top1.avg, duration))

    return objs.avg, top1.avg



if __name__ == '__main__':
    main()
