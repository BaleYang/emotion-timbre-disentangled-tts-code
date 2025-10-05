import os
import json

import torch
import numpy as np

from model import ScheduledOptim
import model as model_pkg

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    # Map friendly names to actual classes exported by model_pkg
    name_map = {
        'Model': getattr(model_pkg, 'Model'),
        'FS2': getattr(model_pkg, 'FS2', None),
    }
    model_name = model_config.get('model_name', 'Model')
    print(f"model name: {model_name}")

    if model_name in name_map and name_map[model_name] is not None:
        model_cls = name_map[model_name]
    else:
        # Fallback to getattr for any legacy names present in the package
        model_cls = getattr(model_pkg, model_name)

    model = model_cls(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            './output',
            str(args.version),
            'ckpt',
            f"{args.restore_step}.pth.tar",
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_mist(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = MIST(preprocess_config, model_config).to(device)
    
    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, 0
        )
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

