import pytorch_lightning as pl
import torch
import torchmetrics
from torchvision import transforms as transforms


class LitModule(pl.LightningModule):
    # ネットワークモジュールなどの定義
    def __init__(self, config, model=None, loss_fn=None):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = self.config.lr
        self.init_metrics()



    def init_metrics(self):
        self.dice = torchmetrics.Dice()

    # オプティマイザの定義
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = self.config.max_epoch,
                eta_min = self.config.lr_min,
                last_epoch = -1
            )

        # lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ==================================================================
    def forward(self, batch):
        imgs = batch
        preds = self.model(imgs)
        return preds

    def training_step(self, batch, batch_idx):
        preds, loss, dice = self._shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_dice", dice, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, dice = self._shared_step(batch)

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True, logger=True)

        return loss


    def _shared_step(self, batch):
        imgs, labels = batch
        preds = self.model(imgs)
        if self.config.img_size != 256:
            preds = torch.nn.functional.interpolate(preds, size=256, mode='bilinear')
        #     labels = torch.permute(labels, (0,3,2,1))
        #     labels = torch.nn.functional.interpolate(labels, size=256, mode='nearest')
        loss = self.loss_fn(preds, labels)
        dice = self.dice(preds, labels.long())
        
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        

        return preds, loss, dice
