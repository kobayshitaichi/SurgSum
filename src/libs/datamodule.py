import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import pandas as pd

from .dataset import ExtractorDataset

from logging import getLogger


import pytorch_lightning as pl
from torch.utils.data import DataLoader

__all__ = ["datamodule"]

logger = getLogger(__name__)


class ExtractorDataModule(pl.LightningDataModule):
    def __init__(self, df, config):
        super().__init__()
        self.config = config
        self.img_size = self.config.img_size
        self.df = df

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = ExtractorDataset(df=self.df, stage="train", config = self.config)
            self.val_data = ExtractorDataset(df=self.df, stage="validation", config = self.config)
        
        logger.info("datamodule setup done")


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.config.batch_size, 
            num_workers=4, 
            shuffle=False,
            pin_memory=True
        )

