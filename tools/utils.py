import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import time
from sklearn.metrics import roc_curve
from thop import profile
import copy

from .multadds_count import add_flops_counting_methods

INIT_TIMES = 100
LAT_TIMES  = 1000


def measure_multadds_fw(model, input_shape, is_cuda=True):

    if len(input_shape) == 3:
        input_data = torch.randn((1,) + tuple(input_shape))
    else:
        input_data = torch.randn(tuple(input_shape))
    if is_cuda:
        input_data = input_data.cuda()

    model.eval()

    model = add_flops_counting_methods(model)
    if is_cuda:
        model = model.cuda()
    model.start_flops_count()
    with torch.no_grad():
        output_data = model(input_data)

    mult_adds = model.compute_average_flops_cost() / 1e6    # M
    return mult_adds


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_2cls(output, target):
    batch_size = target.size(0)
    assert batch_size == output.size(0)

    _, prediction = torch.max(output.data, 1)
    accuracy = prediction.eq(target.data).sum()*100.0/(batch_size)
    return accuracy


def alaska_weighted_auc(y_true, y_pred):
    y_true = np.array(y_true)
    fpr, tpr, thresholds = roc_curve((y_true > 0).astype(int), y_pred, drop_intermediate=False)
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2.0, 1.0]
    auc_x = 0.0
    for idx in range(len(tpr_thresholds) - 1):
        mask = tpr >= tpr_thresholds[idx]
        x = fpr[mask]
        y = tpr[mask]
        mask = y > tpr_thresholds[idx + 1]
        y[mask] = tpr_thresholds[idx + 1]
        y = y - tpr_thresholds[idx]
        auc_x = auc_x + weights[idx] * np.trapz(y, x)
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, np.array(weights))
    return auc_x / normalization


from bisect import bisect_left

class tf_roc():
    def __init__(self, predicts, labels, threshold_num, predict_label_file=None):
        '''predict_score,label
        the predict_score should be between 0 and 1
        the label should be 0 or 1
        threshold_num: number of threshold will plot'''

        self.predicts = []
        self.labels = []
        self.total = 0

        if predict_label_file is not None:
            fd = open(predict_label_file)
            fdl = fd.readline()
            while len(fdl) > 0:
                fdl = fdl.replace('\n', '')
                val = fdl.split(',')
                # val[2] = val[2].split('\\')[0]
                self.predicts.append(float(val[1]))
                self.labels.append(True if int(eval(val[2])) == 1 else False)
                fdl = fd.readline()
                self.total += 1
            fd.close()
        else:
            if not isinstance(predicts, list):
                predicts = list(predicts)
            if not isinstance(labels, list):
                labels = list(labels)
            self.predicts = predicts
            self.labels = labels
            self.total = len(self.labels)
        print(self.total)
        self.threshold_num = threshold_num
        self.trues = 0  # total of True labels
        self.fpr = []  # false positive rate
        self.tpr = []  # true positive rate
        self.ths = []  # thresholds
        self.tn = []  # true negative
        self.tp = []  # true positive
        self.fp = []  # false positive
        self.fn = []  # false negative
        self.calc()

    def calc(self):
        for label in self.labels:
            if label:
                self.trues += 1
        # print 'self.trues:', self.trues
        threshold_step = 1. / self.threshold_num
        for t in range(self.threshold_num + 1):
            th = 1 - threshold_step * t
            tn, tp, fp, fn, fpr, tpr = self._calc_once(th)
            self.fpr.append(fpr)
            self.tpr.append(tpr)
            self.ths.append(th)
            self.tn.append(tn)
            self.tp.append(tp)
            self.fp.append(fp)
            self.fn.append(fn)

    def _calc_once(self, t):
        fp = 0
        tp = 0
        tn = 0
        fn = 0
        # print 't:', t
        for i in range(self.total):
            # print 'labels:', self.labels[i], ' predicts:', self.predicts[i]
            if not self.labels[i]:  # when labels[i] == 0 or false
                if self.predicts[i] >= t:
                    fp += 1  # false positive
                    # print 'fp == ', 'labels:', self.labels[i], ' predicts:', self.predicts[i]
                else:
                    tn += 1  # true negative
                    # print 'tn == ', 'labels:', self.labels[i], ' predicts:', self.predicts[i]
            elif self.labels[i]:
                if self.predicts[i] >= t:  # when labels[i] == 1 or true
                    tp += 1  # true positive
                    # print 'tp == ', 'labels:', self.labels[i], ' predicts:', self.predicts[i]
                else:
                    fn += 1  # false negative
                    # print 'fn == ', 'labels:', self.labels[i], ' predicts:', self.predicts[i]
        fpr = fp / float(fp + tn)  # precision
        tpr = tp / float(self.trues)

        return tn, tp, fp, fn, fpr, tpr

    def get_pmd(self, thresh):
        # thresh is pre-defined False Positive Rate (FPR)
        tpr = self.tpr
        fpr = self.fpr

        #  fnr = 1 - self.tpr

        if (thresh >= fpr[-1]):
            print('warning !!!')
            return 1 - tpr[-1]
        elif thresh <= fpr[0]:
            print('warning !!!')
            return 1 - tpr[0]
        pos = bisect_left(fpr, thresh)
        part_fpr = [fpr[pos - 1], fpr[pos]]
        part_pos = [pos - 1, pos]
        part_tpr = [tpr[pos - 1], tpr[pos]]
        if thresh in part_fpr:
            sp_pos = part_pos[part_fpr.index(thresh)]
            pmd = 1.0 - tpr[sp_pos]
        else:
            assert thresh in [0.05], "type error get P_MD(%f)" % (thresh)
            f1 = np.polyfit(part_fpr, part_tpr, 1)
            pmd = 1 - np.polyval(f1, thresh)
        # print('%.2f part_tpr: %s' % (thresh, part_tpr))
        # print('%.2f part_fpr: %s' % (thresh, part_fpr))

        return pmd

    def get_pfa(self, thresh):
        # args: thresh is pre-defined TPR

        tpr = self.tpr
        fpr = self.fpr

        if (thresh >= tpr[-1]):
            print('warning !!!')
            return fpr[-1]
        elif thresh <= tpr[0]:
            print('warning !!!')
            return fpr[0]
        pos = bisect_left(tpr, thresh)
        part_tpr = [tpr[pos - 1], tpr[pos]]
        part_pos = [pos - 1, pos]
        part_fpr = [fpr[pos - 1], fpr[pos]]
        if thresh in part_tpr:
            sp_pos = part_pos[part_tpr.index(thresh)]
            pfa = fpr[sp_pos]
        else:
            # pfa = get_FA(part_tpr, part_fpr, thresh)
            assert thresh in [0.3, 0.5, 0.7], "type error get P_FA(%f)" % (thresh)
            # p1 = np.poly1d(f1)

            f1 = np.polyfit(part_tpr, part_fpr, 1)
            pfa = np.polyval(f1, thresh)

        print('%.2f part_tpr: %s' % (thresh, part_tpr))
        print('%.2f part_fpr: %s' % (thresh, part_fpr))
        # print ('P_FA(%f): %f'%(thresh, pfa))

        return pfa


def drop_connect(x, training=False, drop_connect_rate=0.0):
    """Apply drop connect."""
    if not training:
        return x
    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def channel_shuffle(x, groups):
    assert groups > 1
    batchsize, num_channels, height, width = x.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: {}'.format(kernel_size)
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2

""" Network profiling """
def get_net_device(net):
    return net.parameters().__next__().device

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6

def count_params_flops(net, input_size):
    # e.g. img_size=(1, 3, 256, 256)
    if isinstance(net, nn.DataParallel):
        net = net.module
    print('input_size: ', input_size)
    original_device = get_net_device(net)

    # print(net)
    inputs = torch.randn(input_size).to(original_device)
    flops, params = profile(copy.deepcopy(net), inputs=(inputs,))
    # print('Total params: %.2fM' % (params/1000000.0))
    # print('Total flops: %.2fM' % (flops/1000000.0))
    return params/1e6, flops/1e6

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))
    
    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
