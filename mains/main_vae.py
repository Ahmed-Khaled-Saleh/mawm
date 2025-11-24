from fastcore import *
from fastcore.utils import *
import torch


import argparse
from os.path import join, exists
from os import mkdir
import time
import os
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import ConcatDataset
from omegaconf import OmegaConf
from dotenv import load_dotenv



from MAWM.core import get_cls

from MAWM.data.utils import transform_train, transform_test, BufferAwareConcatDataset
from MAWM.data.loaders import RolloutObservationDataset

from MAWM.optimizer.utils import ReduceLROnPlateau, EarlyStopper
from MAWM.trainers.vae_trainer import VAETrainer
from MAWM.writers.wandb_writer import WandbWriter


# cfg = OmegaConf.load(join("../cfgs", "vae", "cfg.yaml"))


parser = argparse.ArgumentParser(description='VAE Training')
parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
parser.add_argument('--timestamp', type=str, help='Time stamp', required=True)
parser.add_argument('--env_file', type=str, help='Path to the .env file', required=False)


parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--log_dir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')


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

# cfg.optimizer.lr = float(args.lr) if args.lr else cfg.optimizer.lr
# cfg.data.batch_size = int(args.batch_size) if args.batch_size else cfg.data.batch_size
# cfg.optimizer.name = args.optimizer if args.optimizer else cfg.optimizer.name

def init_data(agent_id):
    train_ds = RolloutObservationDataset(agent= agent_id,
                                        root= cfg.data.data_dir,
                                        transform= transform_train,
                                        buffer_size= cfg.data.buffer_size,
                                        train=True,
                                        obs_key= cfg.data.obs_key
                                        )
    
    test_ds = RolloutObservationDataset(agent= agent_id,
                                        root= cfg.data.data_dir,
                                        transform= transform_test,
                                        buffer_size= cfg.data.buffer_size,
                                        train=False,
                                        obs_key= cfg.data.obs_key
                                        )
    return train_ds, test_ds
    


def init_model():
    model_cls = get_cls(f"MAWM.models.{cfg.model.name.lower()}", cfg.model.name)
    model = model_cls(cfg.model.channels, cfg.model.latent_size)
    return model


def init_opt(model):
    optimizer_cls = get_cls("torch.optim", cfg.optimizer.name)
    optimizer = optimizer_cls(model.parameters(), lr=cfg.optimizer.lr)
    return optimizer

def main(cfg):

    cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    # Fix numeric divergence due to bug in Cudnn
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if cuda else "cpu")

    dss_train, dss_test = [], []
    for agent in cfg.env.agents:
        train_ds, test_ds = init_data(agent)
        dss_train.append(train_ds)
        dss_test.append(test_ds)


    dataset_train = BufferAwareConcatDataset(datasets= dss_train)
    dataset_test = BufferAwareConcatDataset(datasets= dss_test)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=cfg.data.batch_size, shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.data.batch_size, shuffle=True, num_workers=2)

    

    model = init_model()
    model = model.to(device)

    optimizer = init_opt(model= model)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopper(patience=30, min_delta=10)


    def criterion(recon_x, x, mu, logsigma):
        """ VAE loss function """
        BCE = F.mse_loss(recon_x, x, reduction="sum")
        KL = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
        return BCE + KL

    
    # now = time.strftime("%Y%m%d-%H%M%S")
    # cfg.now = now
    writer = WandbWriter(cfg)
    trainer = VAETrainer(cfg, model, train_loader, val_loader, criterion, 
                        optimizer, device,earlystopping, scheduler, writer)

    df_res = trainer.fit()
    df_res.to_csv(join(cfg.log_dir, f"vae_train_val_loss_{cfg.now}.csv"))
    return df_res

if __name__ == "__main__":
    main(cfg)