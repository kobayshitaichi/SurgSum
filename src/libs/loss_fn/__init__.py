from logging import getLogger

import torch.nn as nn

__all__ = ["get_criterion"]
logger = getLogger(__name__)


def get_criterion(
    config,
    loss_fn: str = "dice_loss",
) -> nn.Module:
    if loss_fn == "dice_loss":
        criterion = 0

    else:
        message = "loss function not found"
        logger.error(message)
        print(loss_fn)
        raise ValueError(message)
    return criterion


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, preds, targets):
        ce_loss = nn.CrossEntropyLoss()(preds, targets)
        return ce_loss
