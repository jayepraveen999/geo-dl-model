import argparse
import os
from datetime import datetime
import kornia.augmentation as K
import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchgeo.transforms import AugmentationSequential

from dataset import H8Dataset
from trainer import UNetTrainer, BestMetricCheckpoint
from utils.helper_functions import get_input_channels_count, save_experiment_logs
import warnings

# Suppress all warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(args):
    """Run train and tests loops after defining the environmetal variables,
    datasets, dataloaders, and model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    pl.seed_everything(args.seed)
    torch.use_deterministic_algorithms(args.use_deterministic_algorithms)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.normalize_mwir_lwir_bands:
        if args.bands != [0, 4]:
            raise ValueError("Only MWIR and LWIR bands can be normalized")

    train_transforms_gpu = AugmentationSequential(
        K.RandomHorizontalFlip(p=0.3, keepdim=True),
        K.RandomVerticalFlip(p=0.3, keepdim=True),
        data_keys=["image", "mask"],
    ).to(device)

    train_dataset = H8Dataset(
        data_dir=args.train_data_dir,
        static_data_dir=args.static_data_dir,
        split="train",
        time_steps=args.time_steps,
        bands=args.bands,
        aux_data=args.aux_data,
        transforms=train_transforms_gpu if args.train_transforms else None,
        normalize_bands=args.normalize_mwir_lwir_bands,
    )
    val_dataset = H8Dataset(
        data_dir=args.train_data_dir,
        static_data_dir=args.static_data_dir,
        split="val",
        time_steps=args.time_steps,
        bands=args.bands,
        aux_data=args.aux_data,
        transforms=train_transforms_gpu if args.train_transforms else None,
        normalize_bands=args.normalize_mwir_lwir_bands,

    )
    test_dataset = H8Dataset(
        data_dir=args.test_data_dir,
        static_data_dir=args.static_data_dir,
        split="test",
        time_steps=args.time_steps,
        bands=args.bands,
        aux_data=args.aux_data,
        normalize_bands=args.normalize_mwir_lwir_bands,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # change to True when sampler = None
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=os.cpu_count(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=os.cpu_count(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=os.cpu_count(),
    )

    early_stop_callback = EarlyStopping(
        monitor="val_iou",
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode="max",
    )

    logname = (
        f"{args.exp_name}_{args.batch_size}" + "{epoch:02d}-{val_iou:.2f}"
    )  # noqa: E501

    tb_logger = TensorBoardLogger(save_dir=LIGHTNING_LOGS_DIR, name="lightning_logs", version=args.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        dirpath=WEIGHTS_LOGS_DIR,
        filename=logname,
        save_top_k=1,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval=None)
    
    best_metric_logger = BestMetricCheckpoint(monitor="val_iou", mode="max")

    input_channels = get_input_channels_count(
        args.time_steps, args.bands, args.aux_data, args.normalize_mwir_lwir_bands
    )
    model = UNetTrainer(
        model_type="UNet2D" if args.time_steps == 1 else "UNet3D",
        input_ch=input_channels,
        enc_ch=args.encoder_ch,
        use_act=args.use_act,
        lr=args.lr,
        tb_log_train_val_images=args.train_val_logging_images,
        t_max=args.t_max,
        pos_weight=args.pos_weight,
        normalize_mwir_lwir_bands = args.normalize_mwir_lwir_bands,
        exp_name = args.exp_name,
        weight_decay = args.weight_decay
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        # precision="16-mixed",
        devices=[0],
        enable_progress_bar=True,
        logger=tb_logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[
                    checkpoint_callback,
                    early_stop_callback,
                    lr_monitor,
                    best_metric_logger
                    ],
        # limit_train_batches=int(1),
        # limit_val_batches=int(1),
    )

    if args.best_model_on_test_set:
        # No training involved. model uses the best performing model (bpm) checkpoint and saves the predictions to the folder
        trainer.test(model=model, dataloaders = test_loader, ckpt_path=args.bpm_ckpt_path)
        save_experiment_logs(args, best_metric_logger, WEIGHTS_LOGS_DIR)

        return
    else:
        trainer.fit(model, train_loader, val_loader)
        print("Best model path: ", checkpoint_callback.best_model_path)

        np.random.seed(args.seed)
        trainer.test(model=model, dataloaders = test_loader)
    
        # for logging the metrics of test dataset and current configuration of the model (arguments) for this run 
        save_experiment_logs(args, best_metric_logger, WEIGHTS_LOGS_DIR)


if __name__ == "__main__":

    # Get current timestamp
    CURRENT_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    # check if logs_weights directory exists, if not create it
    if not os.path.exists(os.path.join(os.getcwd(), 'weights_and_logs')):
        os.mkdir(os.path.join(os.getcwd(), 'weights_and_logs'))

    LIGHTNING_LOGS_DIR = os.path.join(os.getcwd(), 'weights_and_logs')
    WEIGHTS_LOGS_DIR = os.path.join(os.getcwd(), 'weights_and_logs', CURRENT_TIMESTAMP)

    parser = argparse.ArgumentParser(description="Training")

    # Add your arguments
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        type=str,
        help="CUDA Visible Devices",
    )
    parser.add_argument(
        "--train-data-dir",
        default="data/train_test_data/subset_training_dynamic_data.h5",
        type=str,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--test-data-dir",
        default="data/train_test_data/subset_testing_dynamic_data.h5",
        type=str,
        help="Path to test data directory",
    )
    parser.add_argument(
        "--static-data-dir",
        default="data/auxiliary_data",
        type=str,
        help="Path to static data directory",
    )

    parser.add_argument(
        "--time_steps",
        default=1,
        type=int,
        help="Number of time steps. Value should be 1 or 4. For master thesis, we use 1 and not time series data.",
    )

    parser.add_argument("--bands", default=[0,4], type=list, help="List of bands [0, 1, 2, 3, 4, 5]")

    parser.add_argument(
        "--aux_data", default=None, type=list, help='List of auxillary data ["DEM", "LANDCOVER", "BIOMES", "LOCATION", "TIME"]'
    )

    parser.add_argument(
        "--input-ch",
        default=2,
        type=int,
        help="Number of input channels",  #:XXX: Remove this
    )
    parser.add_argument(
        "--encoder-ch",
        default=(32, 64, 128, 256, 512, 1024, 2048),
        type=int,
        help="Encoder channels",
    )
    parser.add_argument("--train-transforms", default=False, type=bool, help="List of train transforms")
    parser.add_argument("--use-deterministic-algorithms", default=True, type=bool, help="Use deterministic algorithms")
    parser.add_argument("--normalize-mwir-lwir-bands", default=False, type=bool, help="Normalize MWIR and LWIR bands and create a single band of it")
    parser.add_argument("--use-act", default=None, type=int, help="Activation function")
    parser.add_argument("--lr", default=1e-3, type=int, help="Learning rate")
    parser.add_argument("--t-max", default=168, type=int, help="t_max for consineannealing LR") #defaults to 168
    parser.add_argument("--pos-weight", default=None, type=int, help="Positive weight for BCEWithLogitsLoss") #Pos_weight is 3593 for WRS (pos_weight=torch.tensor([3593]))
    parser.add_argument("--weight-decay", default=1e-6, type=float, help="L2 Regularization: Weight decay for optimizer") #defaults to 0.0 but use 1e-5 for L2 regularization
    parser.add_argument("--best-model-on-test-set", default=True, type=bool, help="Test the best model on test set after training")
    parser.add_argument("--bpm-ckpt-path", default="bps_weight_file/2024-05-25_14-18_32epoch=319-val_iou=0.24.ckpt", type=str, help="Path to the best model checkpoint")    



    parser.add_argument(
        "--train_val_logging_images", default=[10,5], type=int, help="Logs a random image for a every 20 batches to tensorboard"
    )
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument(
        "--batch-size", default=32, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--num-epochs", default=1000, type=int, help="Total number of epochs"
    )
    parser.add_argument(
        "--patience", default=100, type=int, help="Patience for early stopping"
    )
    parser.add_argument(
        "--exp-name", default=str(CURRENT_TIMESTAMP), type=str, help="Experiment name"
    )

    args = parser.parse_args()
    main(args)
