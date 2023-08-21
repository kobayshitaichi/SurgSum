import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torchvision import transforms as transforms
from logging import getLogger
import numpy as np

__all__ = ["lightningmodule"]

logger = getLogger(__name__)


class ExtractorLitModule(pl.LightningModule):
    # ネットワークモジュールなどの定義
    def __init__(self, config, model=None, loss_fn=None):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = self.config.lr
        self.features = np.zeros((0,2048))
        self.preds = np.zeros((0))
        self.init_metrics()



    def init_metrics(self):
        self.acc_phase = torchmetrics.Accuracy(task='multiclass',num_classes=self.config.out_features)
        self.f1_phase = torchmetrics.F1Score(num_classes=self.config.out_features,task='multiclass',average='macro')

    # オプティマイザの定義
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = self.config.max_epoch,
                eta_min = self.config.lr_min,
                last_epoch = -1
            )

        # lr_scheduler_dict = {"scheduler": scheduler, "intervalf1"step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ==================================================================
    def forward(self, batch):
        imgs = batch
        stem, preds = self.model(imgs)
        return stem, preds

    def training_step(self, batch, batch_idx):
        preds, loss, acc, f1 = self._shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, logger=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True, logger=True)
        
        if batch_idx % 100 == 0:
            logger.info(f'train_acc {batch_idx}: {acc}')
            logger.info(f'train_f1 {batch_idx}: {f1}')

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, acc, f1 = self._shared_step(batch)

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, logger=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, logger=True)
        
        if batch_idx % 50 == 0:
            logger.info(f'val_acc {batch_idx}: {acc}')
            logger.info(f'val_f1 {batch_idx}: {f1}')

        return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, labels = batch
            stem, preds = self.model(imgs)
            preds = F.softmax(preds)
            preds = torch.argmax(preds, dim=1)
        logger.info(self.features.shape)
        logger.info(self.preds.shape)
        self.features = np.concatenate([self.features,stem.cpu().detach().numpy()],0)
        self.preds = np.concatenate([self.preds,np.asarray(preds.cpu()).squeeze()],0)


    def _shared_step(self, batch):
        imgs, labels = batch
        stem, preds = self.model(imgs)
        if self.config.loss_fn == 'ib_focal':
            loss = self.loss_fn(preds, labels, stem)
        else:
            loss = self.loss_fn(preds, labels)
        acc = self.acc_phase(preds, labels)
        f1 = self.f1_phase(preds, labels)
        
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        

        return preds, loss, acc, f1


class SumLitModule(pl.LightningModule):
    # ネットワークモジュールなどの定義
    def __init__(self, config, model=None, loss_fn=None):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = self.config.lr
        self.init_metrics()

    def init_metrics(self):
        self.mse = torchmetrics.MeanSquaredError()

    # オプティマイザの定義
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = self.config.max_epoch,
                eta_min = self.config.lr_min,
                last_epoch = -1
            )

        # lr_scheduler_dict = {"scheduler": scheduler, "intervalf1"step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ==================================================================
    def forward(self, batch):
        feats = batch
        output, weights = self.model(feats)
        return output, weights

    def training_step(self, batch, batch_idx):
        loss, mse = self._shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_mse", mse, on_step=False, on_epoch=True, logger=True)
        logger.info(f'train_mse {batch_idx}: {mse}')

        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse = self._shared_step(batch)

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True, logger=True)
        logger.info(f'val_mse {batch_idx}: {mse}')

        return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            feats, gts = batch
            outputs, weight = self.model(feats)



    def _shared_step(self, batch):
        feats, gts = batch
        outputs, weight = self.model(feats.squeeze(0))
        loss = self.loss_fn(outputs, gts)
        mse = self.mse(outputs,gts)
        
        
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        
        
        return loss, mse
