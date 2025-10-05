import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_param_num
from utils.tools import to_device, log
from model import FastSpeech2LossStage1
from dataset import DatasetESDNeutral
from evaluate import evaluatestage1

import random
import numpy as np

import datetime


"""
Stage 1 training script: Train a vanilla FastSpeech2 (FS2) on ESD Neutral only.
Outputs are organized under ./output/<timestamp_version>/.
"""


def write_dict_to_yaml(data: dict, file_path: str):
    """Dump a python dict to a YAML file for reproducibility."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, allow_unicode=True, sort_keys=False)
        print(f"Saved YAML to {file_path}")
    except Exception as e:
        print(f"Failed to write YAML: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int):
    """Set a global random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_global_seed(seed)

def main(args, configs):
    print("Preparing Stage 1 training (FS2 on ESD Neutral)...")
    
    
    preprocess_config, model_config, train_config = configs

    version_name = datetime.datetime.now().strftime("%y%m%d%H%M")+'_' + model_config.get('version_name', '')
    dir_path = os.path.join('./output', version_name)
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'log'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'results'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'config'), exist_ok=True)
    write_dict_to_yaml(preprocess_config, os.path.join(dir_path, 'config', 'preprocess.yaml'))
    write_dict_to_yaml(model_config, os.path.join(dir_path, 'config', 'model.yaml'))
    write_dict_to_yaml(train_config, os.path.join(dir_path, 'config', 'train.yaml'))
    
    # Get dataset (ESD Neutral only)
    dataset = DatasetESDNeutral(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model: enforce FS2 for Stage 1
    model_config['model_name'] = 'FS2'
    model, optimizer = get_model(args, (preprocess_config, model_config, train_config), device, train=True)

    # model = nn.DataParallel(model)

    num_param = get_param_num(model)
    
    Loss = FastSpeech2LossStage1(preprocess_config, model_config).to(device)
    print("Number of FS2 Parameters:", num_param)


    # Init logger
    # for p in train_config["path"].values():
    #     os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(dir_path, 'log', "train")
    val_log_path = os.path.join(dir_path, 'log', "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()


    
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss

                # print(check_tensors_for_nan(batch[2:]))
                # print(check_tensors_for_nan(output))
                
                losses = Loss(batch, output)
                total_loss = losses[0]
            
                # Backward
                total_loss = total_loss / grad_acc_step

                if step == 1:
                    print(total_loss.item())

                total_loss.backward()

                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    # log(train_logger, step, losses=losses)

                if step % val_step == 0:
                    model.eval()
                    message = evaluatestage1(model, step, (preprocess_config, model_config, train_config))
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            dir_path, 
                            'ckpt',
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )

    parser.add_argument(
        "-v", "--version", type=str, default=0, help="path to model"
    )

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
