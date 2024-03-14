import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import pandas as pd

from .dataset import ExtractorDataset, ASFDataset, SumDataset

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
            self.val_data = ExtractorDataset(df=self.df, stage="val", config = self.config)
            self.test_data = ExtractorDataset(df=self.df, stage="test", config = self.config)
            self.phase_weight = self.train_data.phase_weights
            
        logger.info(f"weight: {self.phase_weight}")
        logger.info(f"train data: {self.train_data.__len__()}")
        logger.info(f"val data: {self.val_data.__len__()}")
        logger.info(f"test data: {self.test_data.__len__()}")
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
        
    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.config.batch_size, 
            num_workers=1, 
            shuffle=False,
            pin_memory=True
        )

class ASFDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = ASFDataset(stage="train", config = self.config)
            self.val_data = ASFDataset(stage="val", config = self.config)
            self.test_data = ASFDataset(stage="test", config = self.config)

            logger.info(f"train data: {self.train_data.__len__()}")
            logger.info(f"val data: {self.val_data.__len__()}")
            logger.info(f"test data: {self.test_data.__len__()}")
            logger.info("datamodule setup done")


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=1, 
            num_workers=4, 
            shuffle=False,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=1, 
            num_workers=0, 
            shuffle=False,
            pin_memory=True
        )

class SumDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = SumDataset(stage="train", config = self.config)
            self.val_data = SumDataset(stage="val", config = self.config)
            self.test_data = SumDataset(stage="test", config = self.config)

            logger.info(f"train data: {self.train_data.__len__()}")
            logger.info(f"val data: {self.val_data.__len__()}")
            logger.info("datamodule setup done")


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=1, 
            num_workers=4, 
            shuffle=False,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=1, 
            num_workers=4, 
            shuffle=False,
            pin_memory=True
        )
        
