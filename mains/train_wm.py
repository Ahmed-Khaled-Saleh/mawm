
import argparse
from os.path import join
import os
import random
from functools import reduce

from omegaconf import OmegaConf
from dotenv import load_dotenv
from tqdm.auto import tqdm

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.distributed as dist

from fastcore import *
from fastcore.utils import *

from mawm.data.utils import init_data
from mawm.models import init_models
from mawm.optimizers.utils import init_opt
from mawm.optimizers.schedulers import Scheduler
from mawm.trainers.utils import init_trainer

from mawm.loggers.wandb_writer import WandbWriter
from mawm.loggers.base import get_logger

parser = argparse.ArgumentParser(description='Dynamics Training')
parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
parser.add_argument('--timestamp', type=str, help='Time stamp', required=True)
parser.add_argument('--env_file', type=str, help='Path to the .env file', required=False)

args = parser.parse_args()

if args.env_file:
    load_dotenv(args.env_file)
    key = os.getenv("WANDB_API_KEY", None)

    if key:
        os.environ["WANDB_API_KEY"] = key     

try:
    cfg = OmegaConf.load(args.config)
except:
    print("Invalid config file path")

cfg.now = args.timestamp 

def seed_all():
    _GLOBAL_SEED = np.random.randint(0, 10000)
    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

def main(cfg):

    dist.init_process_group(backend='nccl')
    seed_all()
    cfg.distributed = True
    
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    logger = get_logger(__name__, force=True)
    logger.info(f"Initialized process with local_rank: {local_rank}")

    train_loader, dist_sampler = init_data(cfg, distributed= cfg.distributed)  

    print(len(train_loader))
    for batch_idx, data in enumerate(train_loader):
        logger.info(f"Data loader working properly, batch {batch_idx} fetched")
        break
    
    start_epoch = 0
    dist_sampler.set_epoch(start_epoch)

    model = init_models(cfg, device= torch.device(f'cuda:{local_rank}'), distributed= cfg.distributed)

    optimizer = init_opt(cfg, model)

    scheduler = Scheduler(
        schedule=cfg.optimizer.scheduler.name,
        base_lr=cfg.optimizer.lr,
        data_loader=train_loader,
        epochs=cfg.epochs,
        optimizer=optimizer,
        batch_size=cfg.data.batch_size,
    )

    logger.info(f"Starting training... from {local_rank}")
    verbose = dist.get_rank() == 0  # print only on global_rank==0
    if verbose:
        writer = WandbWriter(cfg)
    else:
        writer = None
    
    print(dist_sampler)
    trainer = init_trainer(cfg, model, train_loader, sampler = dist_sampler, optimizer= optimizer,
                           device= local_rank, earlystopping= None, scheduler= scheduler,
                           writer= writer, verbose= verbose, logger = logger)

    df_res = trainer.fit()
    df_res.to_csv(join(cfg.log_dir, f"dmpc_train_val_loss_{cfg.now}.csv"))
    return df_res

if __name__ == "__main__":
    main(cfg)