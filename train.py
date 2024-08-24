"""Train diffusion model
"""
from configparser import ConfigParser
import os
import logging
import sys
import atexit
import argparse
import random
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import wandb
from utils import infer_type, EpochTimer
from diffusion import DiffusionModel
from dataset import DataModule
from net2 import UNet
# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #


# ---------------------------------- CONFIG ---------------------------------- #

config = ConfigParser()

config.read('config.ini')
parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--verbose','-v',action='store_true')
parser.add_argument('--debug', '-d',action='store_true')
parser.add_argument('--disable_wandb', '-dw',action='store_true')
args = parser.parse_args()

VERBOSE = args.verbose
DEBUG = args.debug
DISABLE_WANDB=  args.disable_wandb
WANDB_PROJECT_NAME = config.get('settings','WANDB_PROJECT_NAME')
WANDB_USER = config.get('settings', 'WANDB_USER')
LOG_FILE_NAME = config.get('settings', 'LOG_FILE_NAME')
LOG_FILE = f"./logs/{LOG_FILE_NAME}"

IMG_RES = config.getint('params', 'IMG_RES')
NUM_EPOCHS = config.getint('params', 'NUM_EPOCHS')
BATCH_SIZE = config.getint('params','BATCH_SIZE')
LR = config.getfloat('params', 'LR')

# ------------------------------- LOGGING SETUP ------------------------------ #
if DISABLE_WANDB:
    os.environ["WANDB_DISABLED"] = "true"

# Creating log folder and file if not exist
if not os.path.isdir('./logs'):
    os.makedirs('./logs')
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', encoding='utf8') as file:
        pass

logging.basicConfig(filename=LOG_FILE,
                    level=logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                    filemode='w')

# Remove annoying matplotlib.font_manager and PIL logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
if VERBOSE:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))



wandb.init(
    project=WANDB_PROJECT_NAME,
    config=dict(map(
        lambda item: (item[0], infer_type(item[1])), # infer type of item[1]
        config.items('params'))
    ),
    settings=wandb.Settings(code_dir='.') # Code logging
)
wandb_logger = WandbLogger(log_model="all")

# ----------------------------------- SEED ----------------------------------- #
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
# ---------------------------------- CLEANUP --------------------------------- #
def on_exit() -> None:
    """Save log file
    """
    # -------------------------------- Saving logs ------------------------------- #
    wandb.save(LOG_FILE)
    logging.info('Finished Training!')

atexit.register(on_exit)

# ------------------------------- DATA LOADING ------------------------------- #

datamodule = DataModule(f'{args.data_dir}', batch_size=BATCH_SIZE)

def create_net() -> torch.nn.Module:
    """Create NN to model noise

    Returns:
        nn.Module: NN
    """
    return UNet(hid_channels=64)

model = DiffusionModel(create_net, lr=LR)
model = torch.compile(model)
wandb.watch(model.net)
# model = DiffusionModel.load_from_checkpoint('./artifacts/model-mc7dmdff:v1/model.ckpt')
checkpoint_callback = ModelCheckpoint(save_weights_only=True, every_n_epochs=10, save_last=True)
trainer_config = {
    'logger': wandb_logger,
    # 'check_val_every_n_epoch': 100,
    'callbacks': [EpochTimer(), checkpoint_callback]
}
if DEBUG:
    trainer = L.Trainer(fast_dev_run=True, **trainer_config)
else:
    trainer = L.Trainer(max_epochs=NUM_EPOCHS, **trainer_config)
trainer.fit(model=model, datamodule=datamodule)
