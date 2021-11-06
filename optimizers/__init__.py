import argparse

import torch.optim as optim

from .adabound import AdaBound
from .radam import RAdam

__all__ = ['parse_optimizer', 'supported_optimizers']

optimizer_defaults = {
    'sgd': (optim.SGD, 'SGDM', {
        'lr': 0.1,
        'momentum': 0.9,
        'dampening': 0.0,
        'weight_decay': 1e-3,
        'nesterov': False,
    }),
    'adam': (optim.Adam, 'ADAM', {
        'lr': 0.001,
        'weight_decay': 1e-3,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'amsgrad': False,
    }),
    'adabound': (AdaBound, 'AdaBound', {
        'lr': 0.001,
        'weight_decay': 5e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'final_lr': 0.1,
        'gamma': 1e-3,
        'amsbound': False,
    })
}


def supported_optimizers():
    return list(optimizer_defaults.keys())


def required_length(nargs):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) != nargs:
                msg = 'argument "{}" requires exactly {} arguments'.format(self.dest, nargs)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)

    return RequiredLength


def parse_optim_args(args, default_args):
    parser = argparse.ArgumentParser(description='Optimizer parser')
    for k, v in default_args.items():
        if type(v) == bool:
            kwargs = {'action': 'store_false' if v else 'store_true'}
        elif type(v) == list:
            kwargs = {'type': type(v[0]), 'nargs': '+', 'default': v}
        elif type(v) == tuple:
            kwargs = {'type': type(v[0]), 'nargs': '+', 'action': required_length(len(v)), 'default': v}
        else:
            kwargs = {'type': type(v), 'default': v}
        parser.add_argument('--{}'.format(k), **kwargs)
    opt = parser.parse_args(args)

    opt_params_name = ''
    for k, v in default_args.items():
        if opt.__getattribute__(k) != v:
            param_format = '' if type(v) == bool else '_{}'.format(opt.__getattribute__(k))
            opt_params_name += '_{}{}'.format(k, param_format)

    return opt, opt_params_name


def parse_optimizer(optimizer, optim_args, model_params):
    if optimizer not in optimizer_defaults:
        raise RuntimeError('Optimizer {} is not supported'.format(optimizer))

    optim_func, optim_name, def_params = optimizer_defaults[optimizer]

    optim_opts, opt_name = parse_optim_args(optim_args, def_params)
    opt_name = '{}{}'.format(optim_name, opt_name)
    return optim_func(model_params, **vars(optim_opts)), opt_name
