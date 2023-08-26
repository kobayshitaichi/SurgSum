from logging import getLogger

import torch.nn as nn
import torch

__all__ = ["get_criterion"]
logger = getLogger(__name__)


def get_criterion(
    config,
    weight,
    loss_fn: str = "ce_loss",
) -> nn.Module:
    if loss_fn == "ce_loss":
        criterion = CrossEntropyLoss(weight=weight)
    
    elif loss_fn == "ib_focal":
        criterion = IB_FocalLoss(weight=weight, num_classes=config.out_features)
        
    elif loss_fn == "focal":
        criterion = FocalLoss(weight=weight)
    
    elif loss_fn == "mse":
        criterion = nn.MSELoss()
    
    elif loss_fn == 'mae':
        criterion = nn.L1Loss()
    
    elif loss_fn  == 'asf_loss':
        criterion = ASFLoss(config.out_features)
    

    else:
        message = "loss function not found"
        logger.error(message)
        print(loss_fn)
        raise ValueError(message)
    return criterion


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight
    ):
        super().__init__()
        self.weight = weight

    def forward(self, preds, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(preds, targets)
        return ce_loss
    
class IB_FocalLoss(nn.Module):
    def __init__(self, weight, alpha=100, gamma=2., num_classes=7):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, input, target, features):
        features = torch.sum(torch.abs(features), 1).reshape(-1, 1)
        grads = torch.sum(torch.abs(nn.functional.softmax(input, dim=1) - nn.functional.one_hot(target, self.num_classes)),1) # batch_size * 1
        ib = grads*(features.reshape(-1))
        ib = self.alpha / (ib + self.epsilon)
        return ib_focal_loss(nn.functional.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)       

def ib_focal_loss(input_values, ib, gamma):
    """Computes the ib focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * ib
    return loss.mean() 

class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=2., num_classes=7):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
          
    def forward(self, input, target):
        return focal_loss(input, target, self.weight, self.gamma)       


def focal_loss(input,target,weight,gamma):
    input_values = nn.functional.cross_entropy(input,target,reduction='none',weight=weight)
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class ASFLoss(nn.Module):
    def __init__(self, num_classes=7):
        super(ASFLoss, self).__init__()
        self.n_classes = num_classes  

    def forward(self, input, target, mask):
        loss = 0
        for p in input:
            loss += nn.CrossEntropyLoss(p.transpose(2, 1).contiguous().view(-1, self.n_classes), target.view(-1))
            loss += 0.15 * torch.mean(torch.clamp(
                nn.MSELoss(nn.functional.log_softmax(p[:, :, 1:], dim=1), nn.functional.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                max=16) * mask[:, :, 1:])
        
        return loss