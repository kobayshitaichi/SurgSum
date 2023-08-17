from logging import getLogger

import torch.nn as nn
import timm


__all__ = ["get_model"]

model_names = [
    "resnet50d"
]
logger = getLogger(__name__)


def get_model(config) -> nn.Module:
    name = config.model.lower()

    if name not in model_names:
        message = "There is no model appropriate to your choice. "
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))
    if name == 'resnet50d':
        model = OneHeadResNet50(config)

    return model

class OneHeadResNet50(nn.Module):
    def __init__(self,hparams):
        super(OneHeadResNet50, self).__init__()
        self.backbone = timm.create_model(hparams.model_name,pretrained=hparams.pretrained, num_classes=0)
        self.in_features = self.backbone.num_features
        self.out_features = hparams.out_features
        self.fc_phase = nn.Linear(self.in_features,self.out_features)

    def forward(self,x,train,y_phase):
        out_stem = self.backbone(x)
        phase = self.fc_phase(out_stem)

        return out_stem, phase