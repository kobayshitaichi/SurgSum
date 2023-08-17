from logging import getLogger

import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

__all__ = ["get_dataloader"]

logger = getLogger(__name__)


class ExtractorDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, stage="train"):
        self.stage = stage
        self.df = df
        self.df = self.df[self.df["stage"] == self.stage]
        self.config = config
        self.class_labels = self.get_labels()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        video_name = "video" + str(row.video_idx).zfill(2)
        data_path = os.path.join(self.config.dataset_dir, "video_split", video_name, row.file_name)
        img = Image.open(data_path)
        img = np.array(img)
        if self.stage == 'train':
            img = self.transform()(image=img)["image"]
        label = torch.tensor(self.class_labels[row.phase])

        return img.float(), label.float()
        


    def __len__(self):
        return len(self.df)
    
    def get_labels(self):
        class_labels = {}
        for i,label in enumerate(self.df.phase.unique()):
            class_labels[label] = i
            print(label, i)
        return class_labels

    def transform(self):
        transforms = [
                A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ]
        
        if self.stage == 'train':
            if self.config.aug_ver == 1:
                transforms += [
                A.RandomResizedCrop(always_apply=False, p=1.0, height=self.img_size, width=self.img_size, scale=(0.7, 1.2), ratio=(0.75, 1.3), interpolation=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ]
            elif self.config.aug_ver == 2:
                transforms += [
                    A.HorizontalFlip(p=0.3),
                    A.VerticalFlip(p=0.3),
                ]
            
        transforms.append(ToTensorV2(p=1))

        return A.Compose(transforms)
