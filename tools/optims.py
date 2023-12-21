# -*- coding: utf-8 -*-
# optimizers and lr_scheduler selection

import warnings

import torch


# __all__ == ['get_optimizer', 'get_scheduler']

def get_optimizer(model_params, optim_name, init_lr, optim_params):

    if optim_params is not None:
        optim_params_dict = eval(optim_params)
    else:
        optim_params_dict = {}
    print(optim_params_dict)

    if optim_name == 'adamax':
        optimizer = torch.optim.Adamax(model_params, lr=init_lr, **optim_params_dict)
    elif optim_name == 'sgd':
        nesterov = optim_params_dict.get('nesterov', False)
        optimizer = torch.optim.SGD(model_params, lr=init_lr, **optim_params_dict)
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model_params, lr=init_lr, **optim_params_dict)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=init_lr, **optim_params_dict)
    elif optim_name == 'adadelta':
        optimizer = torch.optim.Adadelta(model_params, lr=init_lr, **optim_params_dict)
    else:
        raise NotImplementedError
    return optimizer


def get_scheduler(optimizer, scheduler_name, scheduler_params_str, args, batches_in_epoch=None):
    assert isinstance(scheduler_params_str, str)

    if scheduler_params_str is not None:
        scheduler_params = eval(scheduler_params_str)
        print(scheduler_params)
    else:
        scheduler_params = {}

    if scheduler_name == 'explr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)

    elif scheduler_name == 'cosinelr':
        '''
        if args.eta_min is not None:
            eta_min = args.eta_min
        else:
            # eta_min = scheduler_params.get('eta_min', 0.05*args.init_lr)
            eta_min = 0
        '''
        if scheduler_params == {}:
            scheduler_params = dict(
                T_max=args.epochs,
                eta_min=0.05*args.init_lr
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_name == 'plateau':
        if scheduler_params == {}:
            scheduler_params = dict(
                mode='min',
                factor=0.5,
                patience=1,
                verbose=False,
                threshold=0.0001,
                threshold_mode='abs',
                cooldown=0,
                min_lr=1e-8,
                eps=1e-08
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

    else:
        raise NotImplementedError

    return scheduler


