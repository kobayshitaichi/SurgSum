from logging import getLogger

import torch.nn as nn
import timm

from .asformer import ASFormer
from .summarizer import PGL_SUM
from .tcn import MultiStageModel


__all__ = ["get_model"]

model_names = [
    "resnet50d",
    "asformer",
    "asformer_2",
    "pgl_sum",
    "mstcn",
]
logger = getLogger(__name__)


def get_model(config) -> nn.Module:
    name = config.model_name.lower()

    if name not in model_names:
        message = "There is no model appropriate to your choice. "
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))
    if name == 'resnet50d':
        model = OneHeadResNet50(config)
        
    elif name == 'asformer':
        model = ASFormer(3,10,2,2,64,2048,config.out_features,config.channel_masking_rate)

    elif name == 'asformer_2':
        model = ASFormer(2,10,2,2,64,2048,config.out_features,config.channel_masking_rate)
        
    elif name == 'pgl_sum':
        model = PGL_SUM(input_size=2048, output_size=2048, pos_enc='absolute', heads=8)
    
    elif name == 'mstcn':
        model = MultiStageModel(num_stages=4, num_layers=10, num_f_maps=64, dim=2048, num_classes=config.out_features)
        
    return model

class OneHeadResNet50(nn.Module):
    def __init__(self,hparams):
        super(OneHeadResNet50, self).__init__()
        self.backbone = timm.create_model(hparams.model_name,pretrained=hparams.pretrained, num_classes=0)
        self.in_features = self.backbone.num_features
        self.out_features = hparams.out_features
        self.fc_phase = nn.Linear(self.in_features,self.out_features)

    def forward(self,x):
        out_stem = self.backbone(x)
        phase = self.fc_phase(out_stem)

        return out_stem, phase