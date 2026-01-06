
import argparse
from os.path import join
import os
import time
from functools import reduce

from omegaconf import OmegaConf
from dotenv import load_dotenv
from tqdm.auto import tqdm
from fastcore import *
from fastcore.utils import *


import torch
import numpy as np
from matplotlib import pyplot as plt

from mawm.logger import Logger, MetricTracker
from mawm.core import get_cls
from mawm.writers.wandb_writer import WandbWriter
from mawm.data.utils import init_data
from mawm.models.jepa import JEPA
from mawm.models.misc import JepaProjector, MsgPred, ObsPred
from mawm.models.vision import SemanticEncoder
from mawm.optimizers.schedulers import Scheduler
from mawm.optimizers.factory import OptimizerFactory, OptimizerType
from mawm.trainers.trainer_dynamics import DynamicsTrainer


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


def init_opt(params):
        optimizer_cls = get_cls("torch.optim", cfg.optimizer.name)
        optimizer = optimizer_cls(params, lr=cfg.optimizer.lr)
        return optimizer

def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.data.data_dir = "../nbs/data_test"
    train_loader, val_loader = init_data(cfg)   
     

    jepa = JEPA(cfg.model, input_dim=(3, 42, 42), action_dim= 1)
    msg_encoder = SemanticEncoder(num_primitives= 5, latent_dim = 32)#cfg.model.msg_encoder.latent_dim)

    z_input_dim = reduce(lambda x, y: x * y, jepa.backbone.repr_dim)
    # projector = JepaProjector(z_input_dim= z_input_dim, c_input_dim=msg_encoder.latent_dim)
    msg_pred = MsgPred(h_dim=32)
    obs_pred = ObsPred(h_dim=32)
    
    model = {
         'jepa': jepa,
         'msg_encoder': msg_encoder,
        #  'projector': projector,
         'msg_predictor': msg_pred,
         'obs_predictor': obs_pred,
    }

    all_params = (
        list(jepa.parameters()) + 
        list(msg_encoder.parameters()) + 
        list(msg_pred.parameters()) +
        list(obs_pred.parameters())
    )

    optimizer = init_opt(all_params)
    scheduler = Scheduler(
                        schedule=cfg.optimizer.scheduler.name,
                        base_lr=cfg.optimizer.lr,
                        data_loader=train_loader,
                        epochs=cfg.epochs,
                        optimizer=optimizer,
                        batch_size=cfg.data.batch_size,
                        )
                    
    # now = time.strftime("%Y%m%d-%H%M%S")
    # cfg.now = now
    writer = WandbWriter(cfg)
    trainer = DynamicsTrainer(cfg, model, train_loader, val_loader, optimizer= optimizer,
                            device= device, earlystopping= None, scheduler= scheduler, writer= writer)

    df_res = trainer.fit()
    df_res.to_csv(join(cfg.log_dir, f"dmpc_train_val_loss_{cfg.now}.csv"))
    return df_res

if __name__ == "__main__":
    main(cfg)