import PIL.BmpImagePlugin
import PIL.GifImagePlugin
import PIL.JpegImagePlugin
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image, ImageSequence
import PIL
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import ToTensor
import torch
from utils import channel_last, channel_first
import torchvision.transforms as T
import pytorch_lightning as L
import numpy as np
import logging

class JPGDataset(Dataset):
    def __init__(self, file_paths, transform, stage):
        super().__init__()
        self.images = file_paths
        self.stage = stage
        self.transform = transform

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        f = self.images[idx]

        with Image.open(f) as pil_image:
            img = pil_to_tensor(pil_image.convert('RGB')) / 255

        img = self.transform(img, self.stage)
        return img

class Transformation:
    def __init__(self, load_dim=286, target_dim=256):
        self.transform_augment = T.Compose([
            T.Resize((load_dim,load_dim)),
            T.RandomCrop((target_dim, target_dim)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
        self.transform = T.Resize((target_dim, target_dim))
    def __call__(self, img, stage):
        if stage == 'fit':
            img = self.transform_augment(img)
        else:
            img = self.transform(img)
        return img * 2 - 1
    
class DataModule(L.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_val_images, loader_config={}):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.loader_config = loader_config
        self.batch_size = batch_size
        self.num_val_images = num_val_images

        file_paths = list(self.root_dir.glob('*.jpg'))
        self.file_paths = list(filter(DataModule.verify_path, file_paths))
        
        logging.info(f"{len(file_paths) - len(self.file_paths)} corrupted images, {len(self.file_paths)} usable images")

        if len(self.file_paths) == 0:
            raise Exception(f"No JPG files found in {root_dir}")
        self.transform = Transformation()

    @staticmethod
    def verify_path(file):
        # Save time
        
        # corrupted_files = [Path('./data/PetImages/Cat/666.jpg')]
        # return file not in corrupted_files

        try:
            with Image.open(file) as pil_image:
                img = pil_to_tensor(pil_image.convert('RGB')) / 255
            
            assert img.shape[0] == 3
        except:
            print(file)
            return False

    def setup(self, stage):
        self.train_dataset = JPGDataset(self.file_paths, self.transform, 'fit')
        self.val_dataset = JPGDataset(self.file_paths, self.transform, None)
        self.val_batch_generator = DataLoader(self.val_dataset, batch_size=self.num_val_images, shuffle=True)
        
    def train_dataloader(self):
        loader_config = {
            'num_workers': 15,
            "shuffle": True,
            "batch_size": self.batch_size,
            **self.loader_config
        }
        loader = DataLoader(self.train_dataset, **loader_config)
        return loader

    def val_dataloader(self):
        loader_config = {
            'num_workers': 15,
            "shuffle": True,
            "batch_size": self.num_val_images,
            **self.loader_config
        }
        loader = DataLoader(self.val_dataset, **loader_config)
        return loader
          
if __name__ == '__main__':
    dm = DataModule('./data/PetImages/Cat', 1, {})
    dm.setup('fit')
    dl = dm.train_dataloader()

    for x in dl:
        plt.figure()
        plt.imshow(channel_last((x[0:1,:,:,:]+1)/2)[0])
        plt.show()
        break
