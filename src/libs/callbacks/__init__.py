from logging import getLogger

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
    Timer,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from .util_callbacks import ImagePredictionLogger

__all__ = ["get_callbacks"]
logger = getLogger(__name__)


def get_callbacks(
    result_path="",
    callbacks: list = [],
) -> None:
    callbacks_all = []
    for callback in callbacks:
        if callback == "early_stop":
            early_stop_callback = EarlyStopping(
                monitor="val_dice",
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode="max",
            )
            callbacks_all.append(early_stop_callback)

        elif callback == "ckpt":
            checkpoint_callback = ModelCheckpoint(
                dirpath=result_path,
                filename="final_model",
                monitor="val_dice",
                save_last=True,
                mode="max",
            )
            callbacks_all.append(checkpoint_callback)

        elif callback == "prog_bar":
            progress_bar = RichProgressBar(
                theme=RichProgressBarTheme(
                    description="green_yellow",
                    progress_bar="green1",
                    progress_bar_finished="green1",
                    batch_progress="green_yellow",
                    time="grey82",
                    processing_speed="grey82",
                    metrics="grey82",
                )
            )
            callbacks_all.append(progress_bar)

        elif callback == "lr_monitor":
            lr_monitor = LearningRateMonitor()
            callbacks_all.append(lr_monitor)

        elif callback == "timer":
            timer = Timer(interval="epoch")
            callbacks_all.append(timer)

        else:
            message = "loss function not found"
            logger.error(message)
            raise ValueError(message)

    return callbacks_all
