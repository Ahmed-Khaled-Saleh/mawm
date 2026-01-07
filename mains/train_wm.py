
import argparse
from os.path import join
import os
import time
import random
from functools import reduce

from omegaconf import OmegaConf
from dotenv import load_dotenv
from tqdm.auto import tqdm
from fastcore import *
from fastcore.utils import *
import torch.multiprocessing as mp

import torch
import numpy as np
from matplotlib import pyplot as plt

from mawm.core import get_cls
from mawm.writers.wandb_writer import WandbWriter
from mawm.optimizers.schedulers import Scheduler
from mawm.logger.base import get_logger
from mawm.trainers.wm_trainer import WMTrainer

from torch.nn.parallel import DistributedDataParallel
from mawm.data.utils import init_data_dist
from mawm.distributed.model_utils import init_models_dist
from mawm.optimizers.utils import init_opt_dis
import torch.distributed as dist



parser = argparse.ArgumentParser(description='Dynamics Training')
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


_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

def main(cfg):

    dist.init_process_group(backend='nccl')

    # seed = cfg.get('seed', _GLOBAL_SEED)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True
    _GLOBAL_SEED = 0
    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    logger = get_logger(__name__, force=True)
    logger.info(f"Initialized process with local_rank: {local_rank}")

    verbose = dist.get_rank() == 0  # print only on global_rank==0

    train_loader, dist_sampler = init_data_dist(cfg)   
    
    start_epoch = 0
    dist_sampler.set_epoch(start_epoch)

    jepa, msg_enc, msg_pred, obs_pred = init_models_dist(cfg, device= torch.device(f'cuda:{local_rank}'))

    jepa = DistributedDataParallel(jepa, device_ids = [local_rank], find_unused_parameters=True)
    msg_enc = DistributedDataParallel(msg_enc, device_ids = [local_rank], find_unused_parameters= True)
    msg_pred = DistributedDataParallel(msg_pred, device_ids = [local_rank], find_unused_parameters=True)
    obs_pred = DistributedDataParallel(obs_pred, device_ids = [local_rank], find_unused_parameters=True)

    optimizer = init_opt_dis(cfg, jepa, msg_enc, msg_pred, obs_pred)
    scheduler = Scheduler(
        schedule=cfg.optimizer.scheduler.name,
        base_lr=cfg.optimizer.lr,
        data_loader=train_loader,
        epochs=cfg.epochs,
        optimizer=optimizer,
        batch_size=cfg.data.batch_size,
    )
    
    model = {
         'jepa': jepa,
         'msg_encoder': msg_enc,
         'msg_predictor': msg_pred,
         'obs_predictor': obs_pred,
    }

    
    writer = WandbWriter(cfg)
    trainer = WMTrainer(cfg, model, train_loader, sampler = dist_sampler, optimizer= optimizer,
                        device= local_rank, earlystopping= None, scheduler= scheduler,
                        writer= writer, verbose= verbose, logger = logger)

    df_res = trainer.fit()
    df_res.to_csv(join(cfg.log_dir, f"dmpc_train_val_loss_{cfg.now}.csv"))
    return df_res

if __name__ == "__main__":
    main(cfg)