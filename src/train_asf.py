import datetime
from logging import INFO, basicConfig, getLogger

import warnings
warnings.simplefilter('ignore')

import os
import numpy as np

from libs.config import get_config
from libs.seed import set_seed
from libs.create_dataframe import get_dataframe
from libs.datamodule import ASFDataModule
from libs.litmodule import ASFLitModule
from libs.callbacks import get_callbacks
from libs.loss_fn import get_criterion
from libs.models import get_model

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

import argparse

logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for /// with /// Dataset.
        """
    )
    parser.add_argument("--config", type=str, help="path of a config file")

    parser.add_argument(
        "--resume",
        required=False,
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )

    parser.add_argument(
        "--debug",
        default=False,
        required=False,
        help="debug",
    )
    
    parser.add_argument(
        "--use_wandb",
        default=False,
        required=False,
        help="Add --use_wandb option if you want to use wandb.",
    )

    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=0,
        help="random seed",
    )
    return parser.parse_args()

def main():
    args = get_arguments()
    set_seed(args.seed)
    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)
    
    # setting logger configuration
    logname = os.path.join(result_path, f"{datetime.datetime.now():%Y-%m-%d}_train.log")
    basicConfig(
        level=INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )
    
    config = get_config(args.config)
    net = get_model(config)
    callbacks = get_callbacks(result_path, config.callbacks, config)
    
    dm = ASFDataModule(config)
    dm.setup()
    loss_fn = get_criterion(loss_fn=config.loss_fn, config=config, weight=None)
    
    lm  = ASFLitModule(config=config, model=net, loss_fn=loss_fn)
    
    if args.use_wandb:
        logger = [WandbLogger(
            project="Summarization",
            job_type="train",
            save_dir=result_path,
            name=experiment_name,
        )]
    else:
        logger = []
    # csv_logger = CSVLogger(save_dir=result_path, name="csvlog")

    trainer = pl.Trainer(
        max_epochs=config.max_epoch,
        min_epochs=config.min_epoch,
        accelerator="gpu",
        devices=[config.devices],
        logger=logger,
        log_every_n_steps=100,
        fast_dev_run=bool(args.debug),
        deterministic=None,  # True
        callbacks=callbacks,
        profiler="simple",
    )

    if config.mode == "fit":
        trainer.fit(model=lm, datamodule=dm)
        
    elif config.mode == "test":
        lm = lm.load_from_checkpoint(os.path.join(result_path,"last.ckpt"),config=config, model=net, loss_fn=loss_fn)
        trainer.test(model=lm, datamodule=dm)
        np.save(os.path.join(result_path,'preds.npy'), lm.preds)

    else:
        trainer.fit(model=lm, datamodule=dm)
        trainer.test(model=lm, datamodule=dm)
        np.save(os.path.join(result_path,'preds.npy'), lm.preds)



if __name__ == "__main__":
    main()
