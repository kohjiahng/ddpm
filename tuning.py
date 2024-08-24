"""Tune diffusion model (find optimal batch size/lr)
"""
import os
from configparser import ConfigParser
import logging
import sys
import argparse
import random
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.callbacks import Timer, ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
import torch
from diffusion import DiffusionModel
from dataset import DataModule
from net2 import UNet
import wandb
# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #


# ---------------------------------- CONFIG ---------------------------------- #

config = ConfigParser()

config.read('config.ini')
parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--verbose','-v',action='store_true')
args = parser.parse_args()

VERBOSE = args.verbose
WANDB_PROJECT_NAME = config.get('settings', 'WANDB_PROJECT_NAME')
IMG_RES = config.getint('params', 'IMG_RES')
NUM_EPOCHS = config.getint('params', 'NUM_EPOCHS')
BATCH_SIZE = config.getint('params','BATCH_SIZE')
LR = config.getfloat('params', 'LR')


# Remove annoying matplotlib.font_manager and PIL logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
if VERBOSE:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

os.environ["WANDB_DISABLED"] = "true"
wandb.init(project=WANDB_PROJECT_NAME)
# ----------------------------------- SEED ----------------------------------- #
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------- DATA LOADING ------------------------------- #

datamodule = DataModule(f'{args.data_dir}', batch_size=BATCH_SIZE)

def create_net() -> torch.nn.Module:
    """Create NN to model noise

    Returns:
        nn.Module: NN
    """
    return UNet(hid_channels=64)

model = torch.compile(DiffusionModel(create_net, lr=LR))
checkpoint_callback = ModelCheckpoint(save_weights_only=True, every_n_epochs=10, save_last=True)
trainer_config = {
    'limit_val_batches': 1,
    # 'check_val_every_n_epoch': 100,
    'callbacks': [Timer(), checkpoint_callback]
}
trainer = L.Trainer(max_epochs=NUM_EPOCHS, **trainer_config)
tuner = Tuner(trainer)
tuner.lr_find(model, datamodule=datamodule)
tuner.scale_batch_size(model, datamodule=datamodule, mode='binsearch')
