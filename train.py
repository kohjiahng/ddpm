import logging
from configparser import ConfigParser
import wandb
from utils import infer_type
import atexit
import os
from diffusion import DiffusionModel
from net import UNet
import sys
import argparse
from dataset import DataModule
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Timer
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
args = parser.parse_args()

VERBOSE = args.verbose
DEBUG = args.debug
WANDB_PROJECT_NAME = config.get('settings','WANDB_PROJECT_NAME')
WANDB_USER = config.get('settings', 'WANDB_USER')
LOG_FILE_NAME = config.get('settings', 'LOG_FILE_NAME')
LOG_FILE = f"./logs/{LOG_FILE_NAME}"

IMG_RES = config.getint('params', 'IMG_RES')
NUM_EPOCHS = config.getint('params', 'NUM_EPOCHS')
BATCH_SIZE = config.getint('params','BATCH_SIZE')
LR = config.getfloat('params', 'LR')

# ------------------------------- LOGGING SETUP ------------------------------ #

# Creating log folder and file if not exist
if not os.path.isdir('./logs'):
    os.makedirs('./logs')
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as file:
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

# ---------------------------------- CLEANUP --------------------------------- #
def on_exit():
    # -------------------------------- Saving logs ------------------------------- #
    if not DEBUG:
        wandb.save(LOG_FILE)

    logging.info('Finished Training!')

atexit.register(on_exit)

# ------------------------------- DATA LOADING ------------------------------- #

datamodule = DataModule(f'{args.data_dir}', num_val_images=2, batch_size=BATCH_SIZE)

model = DiffusionModel(opt_config={'lr': LR})

trainer_config = {
    'limit_val_batches': 1,
    'enable_checkpointing': False,
    'logger': wandb_logger,
    'callbacks': [Timer()]
}
if DEBUG:
    trainer = L.Trainer(fast_dev_run=True, **trainer_config)
else:
    trainer = L.Trainer(max_epochs=NUM_EPOCHS, **trainer_config)
trainer.fit(model=model, datamodule=datamodule)