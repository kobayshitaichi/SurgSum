import datetime
from logging import INFO, basicConfig, getLogger

import warnings
warnings.simplefilter('ignore')

import os

from libs.config import get_config
from libs.seed import set_seed
from libs.create_dataframe import get_dataframe
from libs.datamodule import ExtractorDataModule
from libs.callbacks import get_callbacks
from libs.loss_fn import get_criterion
from libs.models import get_model

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
        "--use_wandb",
        required=False,
        action="store_true",
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
    
    df = get_dataframe(config)
    dm = ExtractorDataModule(df, config)
    dm.setup()
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    