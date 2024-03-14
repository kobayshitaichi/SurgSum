from logging import getLogger

import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils import class_weight
import os
import pandas as pd
import random

__all__ = ["get_dataloader"]

logger = getLogger(__name__)


class ExtractorDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, stage="train"):
        self.stage = stage
        self.df = df
        if self.stage != 'test':
            self.df = self.df[self.df["stage"] == self.stage]
            self.df = self.df[self.df.phase != 'irrelevant']
            self.df = self.df[self.df.phase != 'others']
        else:
            drop_target = self.df.index[(self.df.video_idx>5) &  ((self.df.phase == 'irrelevant') | (self.df.phase == 'others'))]
            self.df = self.df.drop(drop_target).reset_index(drop=True)

        self.config = config
        self.class_labels = self.get_labels()
        self.df['y'] = self.df.phase.map(lambda x: int(self.class_labels[x]))
        if self.stage == 'test':
            self.df['y'] = 1
        y = torch.tensor(list(self.df.y))
        if self.stage != 'test':
            self.phase_weights = torch.tensor(class_weight.compute_class_weight(class_weight ='balanced', 
                                                                classes=np.unique(y), 
                                                                y =y.numpy()), device=self.config.devices,
                                            dtype=torch.float)
        else:
            self.phase_weogjts = torch.ones((6))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        video_name = "video" + str(row.video_idx).zfill(2)
        data_path = os.path.join(self.config.dataset_dir, "video_split", video_name, row.file_name)
        img = Image.open(data_path)
        img = np.array(img)
        img = self.transform()(image=img)["image"]
        label = torch.tensor(row.y)

        return img.float(), label

    def __len__(self):
        return len(self.df)
    
    def get_labels(self):
        class_labels = {}
        for i,label in enumerate(self.df.phase.unique()):
            class_labels[label] = i
            if self.stage == "train":
                logger.info(f"{label}: {i}")
        return class_labels

    def transform(self):
        transforms = [
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                A.Resize(self.config.img_size, self.config.img_size)
        ]
        
        if self.stage == 'train':
            if self.config.aug_ver == 0:
                transforms += []
            elif self.config.aug_ver == 1:
                transforms += [
                A.RandomResizedCrop(always_apply=False, p=0.7, height=self.config.img_size, width=self.config.img_size, scale=(0.7, 1.2), ratio=(0.75, 1.3), interpolation=1),
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                ]
            elif self.config.aug_ver == 2:
                transforms += [
                    A.HorizontalFlip(p=0.3),
                    A.VerticalFlip(p=0.3),
                ]
            elif self.config.aug_ver == 3:
                transforms += [
                    A.RandomResizedCrop(always_apply=False, p=0.7, height=self.config.img_size, width=self.config.img_size, scale=(0.7, 1.2), ratio=(0.75, 1.3), interpolation=1),
                    A.HorizontalFlip(p=0.2),
                    A.VerticalFlip(p=0.2),                
                    A.ShiftScaleRotate(shift_limit=0.1,
                             scale_limit=(-0.2, 0.5),
                             rotate_limit=15,
                             border_mode=0,
                             value=0,
                             p=0.7),
                ]

        transforms.append(ToTensorV2(p=1))

        return A.Compose(transforms)

class ASFDataset(torch.utils.data.Dataset):
    def __init__(self, config, stage="train"):
        self.stage = stage
        self.config = config
        self.data_dir = self.config.feats_dir

        feature_path = os.path.join('../result',self.data_dir)
        self.fe_df = pd.read_csv(os.path.join(feature_path,'processed_df.csv'))
        self.features = np.load(os.path.join(feature_path,'features.npy'))

        action_dict = {
            'design': 0,
            'anesthesia': 1,
            'incision': 2,
            'hemostasis': 3,
            'dissection': 4,
            'closure': 5,
            'others': 6,
            'irrelevant': 7
        }
        
        if self.config.module == 'WR':
            train_vid_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            val_vid_ids = [train_vid_ids.pop(self.config.test_vid_idx)]
            test_vid_ids = val_vid_ids
            self.gts = self.fe_df.phase.map(lambda x: int(action_dict[x])).values
            
        else:
            train_vid_ids = [0,1,2,3,4,5]
            val_vid_ids = [train_vid_ids.pop(self.config.test_vid_idx)]
            test_vid_ids = val_vid_ids
            self.gts = self.fe_df.field.map(lambda x: int(bool(x))).values

        if self.stage=='train':
            self.vid_ids = train_vid_ids
        elif self.stage=='val':
            self.vid_ids = val_vid_ids
        else:
            self.vid_ids = test_vid_ids
            if self.config.module=='WR':
                self.fe_df = self.fe_df[self.fe_df.video_idx==self.vid_ids[0]]
                # self.fe_df['RIF_preds'] = np.load(os.path.join('../result',self.config.RIF_dir,'preds.npy'))
                # self.fe_df = self.fe_df[self.fe_df.RIF_preds==1].reset_index(drop=True)
            
        self.df = self.get_df()   

    def __getitem__(self, index):
        row = self.df.iloc[index]
        start = row.start_idx
        end = row.end_idx

        features = torch.Tensor(self.features[start:end])
        gts = torch.Tensor(self.gts[start:end])
        mask = torch.ones(self.config.out_features,end-start)

        return features, gts.to(torch.int64), mask

    def __len__(self):
        return len(self.df)
    
    def get_df(self):
        df = pd.DataFrame({"video_idx":self.vid_ids})
        start = []
        end = []
        split = []
        for i in sorted(self.vid_ids):
            start.append(self.fe_df[self.fe_df.video_idx==i].index[0])
            end.append(self.fe_df[self.fe_df.video_idx==i].index[-1] + 1)
        df['start_idx'] = start
        df['end_idx'] = end
        return df    

class SumDataset(torch.utils.data.Dataset):
    def __init__(self, config, stage="train"):
        self.stage = stage
        self.config = config
        self.data_dir = self.config.feats_dir
        train_vid_ids = [0,1,2,3,4,5]
        val_vid_ids = [train_vid_ids.pop(self.config.test_vid_idx)]
        feature_path = os.path.join('../result',self.data_dir)
        self.fe_df = pd.read_csv(os.path.join(feature_path,'processed_df.csv'))
        self.features = np.load(os.path.join(feature_path,'features.npy'))
        self.gts = self.fe_df.summary.values
        if self.stage=='train':
            self.vid_ids = train_vid_ids
        else:
            self.vid_ids = val_vid_ids
            
        self.df = self.get_df() 

    def __getitem__(self, index):
        row = self.df.iloc[index]
        start = row.start_idx
        end = row.end_idx
        
        features = torch.Tensor(self.features[start:end])
        gts = torch.Tensor((self.gts[start:end]+1)/2)

        return features, gts

    def __len__(self):
        return len(self.df)
    
    def get_df(self):
        df = pd.DataFrame({"video_idx":self.vid_ids})
        start = []
        end = []
        split = []
        for i in self.vid_ids:
            start.append(self.fe_df[self.fe_df.video_idx==i].index[0])
            end.append(self.fe_df[self.fe_df.video_idx==i].index[-1] + 1)
        df['start_idx'] = start
        df['end_idx'] = end
        
        return df

    def get_labels(self):
        class_labels = {}
        for i,label in enumerate(self.df.phase.unique()):
            class_labels[label] = i
            if self.stage == "train":
                logger.info(f"{label}: {i}")
        return class_labels