"""Datasets and datamodules
"""
import logging
import random
from typing import Callable
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
import torchvision
import torchvision.transforms as T
import pytorch_lightning as L
from utils import channel_last

class Transformation:
    """Data preparation (scaling to [-1,1] + augmentation)
    """
    def __init__(self, load_dim: int = 286, target_dim: int = 256) -> None:
        # self.transform_augment = T.Compose([
        #     T.Resize((load_dim,load_dim)),
        #     T.RandomCrop((target_dim, target_dim)),
        #     T.RandomHorizontalFlip(),
        #     T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        # ])
        self.transform = T.Resize((target_dim, target_dim))
    def __call__(self, img: Tensor, stage: str) -> Tensor:
        """perform transformation on batched images

        Args:
            img (Tensor): 4D tensor of batched images
            stage (str): current stage

        Returns:
            Tensor: 4D tensor of transformed images
        """
        return self.transform(img) * 2 - 1
        # if stage == 'fit':
        #     img = self.transform_augment(img)
        # else:
        #     img = self.transform(img)
        # return img * 2 - 1
class JPGDataset(Dataset):
    """Image dataset
    """
    def __init__(self, file_paths: str, transform: Callable[[Tensor,str], Tensor], stage: str) -> None:
        super().__init__()
        self.images = file_paths
        self.stage = stage
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)
    def __getitem__(self, idx: int) -> Tensor:
        f = self.images[idx]

        imgs = [f'./data/PetImages/Cat/{idx}.jpg' for idx in range(66, 66+100)]
        f = random.choice(imgs)
        with Image.open(f) as pil_image:
            img = pil_to_tensor(pil_image.convert('RGB')) / 255

        img = self.transform(img, self.stage)
        return img


class MNIST(Dataset):
    """MNIST dataset (RGB rescaled to [-1,1])
    """
    def __init__(self) -> None:
        super().__init__()
        transform = T.Compose([
            T.Lambda(lambda x: x.convert('RGB')),
            T.ToTensor(),
            T.Lambda(lambda x: x*2-1)
        ])
        self.dataset = torchvision.datasets.MNIST('./data', download=True, transform=transform)
    def __getitem__(self, index: int) -> Tensor:
        return self.dataset.__getitem__(index)[0]
    def __len__(self):
        return self.dataset.__len__()
class DataModule(L.LightningDataModule):
    """DataModule for diffusion model training
    """
    def __init__(self, root_dir: str | Path, batch_size: int, loader_config: dict | None = None) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.loader_config = (loader_config if loader_config is not None else {})
        self.batch_size = batch_size

        all_file_paths = list(self.root_dir.glob('*.jpg'))
        self.file_paths = list(filter(DataModule.verify_path, all_file_paths))

        # train/val split
        random.shuffle(self.file_paths)
        train_ratio = 0.7
        self.train_paths = self.file_paths[:int(train_ratio*len(self.file_paths))]
        self.val_paths = self.file_paths[int(train_ratio*len(self.file_paths)):]
        if len(self.file_paths) == 0:
            raise ValueError(f"No JPG files found in {root_dir}")

        logging.info("%s corrupted images, %s usable images (%s train %s test)",
                     len(all_file_paths) - len(self.file_paths),
                     len(self.file_paths), len(self.train_paths), len(self.val_paths))

        self.transform = Transformation()
        self.train_dataset = JPGDataset(self.train_paths, self.transform, 'fit')
        self.val_dataset = JPGDataset(self.val_paths, self.transform, None)

    @staticmethod
    def verify_path(file: str) -> bool:
        """Verify image file path

        Args:
            file (str): file path

        Returns:
            bool: whether the file is readable
        """

        # return file in [Path('./data/PetImages/Cat/66.jpg'), Path('./data/PetImages/Cat/67.jpg')]
        # Save time
        corrupted_files = [Path('./data/PetImages/Cat/666.jpg')]
        return file not in corrupted_files

        # try:
        #     with Image.open(file) as pil_image:
        #         img = pil_to_tensor(pil_image.convert('RGB')) / 255
        #     assert img.shape[0] == 3
        #     return True
        # except:
        #     print(file)
        #     return False

    def train_dataloader(self) -> DataLoader:
        loader_config = {
            'num_workers': 15,
            "shuffle": True,
            "batch_size": self.batch_size,
            **self.loader_config
        }
        loader = DataLoader(self.train_dataset, **loader_config)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader_config = {
            'num_workers': 15,
            "shuffle": True,
            "batch_size": self.batch_size,
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
