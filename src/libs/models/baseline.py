import timm
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, num_classes, model_name="", pretrained=True):
        super(Baseline, self).__init__()
        # モデルの定義
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=num_classes
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)
