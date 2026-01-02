from fastcore import *
from fastcore.utils import *


import argparse
from os.path import join, exists
from os import mkdir
import time
import os
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import ConcatDataset
from omegaconf import OmegaConf
from dotenv import load_dotenv



from mawm.core import get_cls, PRIMITIVE_TEMPLATES

from mawm.data.utils import lejepa_train_tf, lejepa_test_tf
from mawm.data.loaders import MarlGridDataset

from mawm.trainers.trainer_progjepa import ProgLejepaTrainer
from mawm.writers.wandb_writer import WandbWriter


import warnings

warnings.filterwarnings("ignore", message="Ill-formed record")

from typing import Optional
import os
import shutil
from dataclasses import dataclass
import dataclasses
import random
import time

import torch
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import matplotlib
from mawm.logger import Logger, MetricTracker

from mawm.optimizers.schedulers import Scheduler, LRSchedule
from mawm.optimizers.factory import OptimizerFactory, OptimizerType
from mawm.trainers.trainer_dynamics import DynamicsTrainer

from mawm.models.jepa import JEPA

parser = argparse.ArgumentParser(description='VAE Training')
parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
parser.add_argument('--timestamp', type=str, help='Time stamp', required=True)
parser.add_argument('--env_file', type=str, help='Path to the .env file', required=False)


args = parser.parse_args()


if args.env_file:
    load_dotenv(args.env_file)
    key = os.getenv("WANDB_API_KEY", None)
    hf_secret = os.getenv("HF_SECRET_CODE", None)

    if key:
        os.environ["WANDB_API_KEY"] = key
    if hf_secret:
        os.environ["HF_SECRET_CODE"] = hf_secret     

try:
    cfg = OmegaConf.load(args.config)
except:
    print("Invalid config file path")

cfg.now = args.timestamp 


def init_data(data_dir, batch_size, train=True):
    train_ds = MarlGridDataset(root=data_dir,
                               transform=lejepa_train_tf,
                               seq_len=cfg.data.seq_len,
                               train=train
                               )
        
    test_ds = MarlGridDataset(data_path= data_dir, 
                              transform= lejepa_test_tf,
                              seq_len= cfg.data.seq_len,
                              train= not train
                              )
        
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader 

def init_opt(model):
        optimizer_cls = get_cls("torch.optim", cfg.optimizer.name)
        optimizer = optimizer_cls(model.parameters(), lr=cfg.optimizer.lr)
        return optimizer

def main(cfg):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    train_loader, val_loader = init_data(cfg.data.data_dir, cfg.data.batch_size, train= True)    

    model = JEPA(cfg.model, action_dim= 1)
    model = model.to(device)
    optimizer = init_opt(model)

    # TODO: Check this
    scheduler = Scheduler(optimizer, 'min', factor=0.5, patience=5) 
    
    # now = time.strftime("%Y%m%d-%H%M%S")
    # cfg.now = now
    writer = WandbWriter(cfg)
    trainer = DynamicsTrainer(cfg, model, train_loader, val_loader, criterion= None, optimizer= optimizer,
                            device= device, earlystopping= None, scheduler= scheduler, writer= writer)

    df_res = trainer.fit()
    df_res.to_csv(join(cfg.log_dir, f"dmpc_train_val_loss_{cfg.now}.csv"))
    return df_res

if __name__ == "__main__":
    main(cfg)