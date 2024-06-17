import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import numpy as np
import yaml
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from model import UNet2D, UNet3D
from pytorch_lightning import Callback
from utils.helper_functions import revert_feature_standardization

# get global mean and std of all bands for training data to standardize the data
config_file = "config/global_mean_std.yaml"

with open(config_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

GLOBAL_BAND_MEANS = config["GLOBAL_MEAN_SUBSET"]
GLOBAL_BAND_STDS = config["GLOBAL_STD_SUBSET"]



class UNetTrainer(pl.LightningModule):
    """Pytorch Lightning triner class for the pixel-wise classification
    task of active fire detection."""

    def __init__(
        self,
        model_type="UNet2D",
        input_ch=1,
        use_act=None,
        enc_ch=(32, 64, 128, 256, 512, 1024),
        lr=1e-3,
        tb_log_train_val_images=[20,20],
        t_max=168,
        pos_weight=None,
        normalize_mwir_lwir_bands=False,
        exp_name = None,
        weight_decay = 0.0,
    ):
        """Initialize the UNetTrainer class.

        Args:
            model_type (str, optional): model type. Defaults to "UNet2D".
            input_ch (int, optional): number of input channels. Defaults to 10.
            use_act (nn.module, optional): activation function. Defaults to None.
            enc_ch (tuple, optional): encoder channels. Defaults to (32, 64, 128, 256, 512, 1024).
            lr (float, optional): learning rate. Defaults to 1e-3.
            tb_log_pred_gt (bool, optional): whether to plot predictions and annotations in tboard. Defaults to False.
            t_max (int, optional): Used for t_max paramter which defines number of iterations (number of batches) for CosineAnnealing LR. Defaults to 168.
            pos_weight (torch.tensor, optional): positive weight for BCEWithLogitsLoss. Defaults to None.
            normalize_mwir_lwir_bands (bool, optional): Defaults to False. If true, skips destandardization as it is not necessary.
            exp_name (str, optional): Experiment name. Defaults to None.
            weight_decay (float, optional): Weight decay. Defaults to 0.0.

        """  # noqa: E501
        super().__init__()

        self.tb_log_train_val_images = tb_log_train_val_images 
        self.training_log_images_idx = 0 # used to log images to tensorboard
        self.validation_log_images_idx = 0 # used to log images to tensorboard
        self.test_log_images_idx = 0 # used to log images to tensorboard

        self.exp_name = exp_name
        self.normalize_mwir_lwir_bands = normalize_mwir_lwir_bands
        self.t_max = t_max 
        self.weight_decay = weight_decay

        self.iou = torchmetrics.JaccardIndex(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.fscore = torchmetrics.F1Score(task="binary")
        self.acc = torchmetrics.classification.Accuracy(task="binary")
        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([int(pos_weight)]) if pos_weight is not None else None)

        if model_type == "UNet2D":
            self.model = UNet2D(
                input_ch=input_ch,
                use_act=use_act,
                encoder_channels=enc_ch,
            )
        else:
            self.model = UNet3D(
                input_ch=input_ch,
                use_act=use_act,
                encoder_channels=enc_ch,
            )

        self.global_band_means_stds = np.column_stack(
            (GLOBAL_BAND_MEANS, GLOBAL_BAND_STDS)
            )[[0]] #XXX: Change this to all input bands if we want to log the images of all bands. Currently restricting to MWIR visualization only
        self.global_mean, self.global_std = list(self.global_band_means_stds[:, 0]), list(
            self.global_band_means_stds[:, 1]
        )

    def forward(self, x):
        """Run forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Define the training step."""
        self.training_log_images_idx += 1
        x, y, idx = batch  # - represents the idx that we return from the dataset
        x = x.to(
            memory_format=torch.channels_last
        ) 
        x = x.float()
        y = y.float()

        preds = self.model.forward(x)
        loss = self.loss(preds, y)

        pred_mask = (preds > 0.5).float()
        iou = self.iou(pred_mask, y)
        precision = self.precision(pred_mask, y)
        recall = self.recall(pred_mask, y)
        fscore = self.fscore(pred_mask, y)
        accuracy = self.acc(pred_mask, y)
        metrics = {
            "train_loss": loss,
            "train_iou": iou,
            "train_precision": precision,
            "train_recall": recall,
            "train_fscore": fscore,
            "train_accuracy": accuracy,
        }
        self.log_dict(metrics, logger=True, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)

        if self.training_log_images_idx % self.tb_log_train_val_images[0]== 0:
            self.log_images(x, y, pred_mask, idx, self.training_log_images_idx, mode="train")

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Define the validation step."""
        self.validation_log_images_idx += 1
        x, y, idx = batch
        x = x.to(memory_format=torch.channels_last)
        x = x.float()
        y = y.float()

        preds = self.model.forward(x)
        loss = self.loss(preds, y)

        pred_mask = (preds > 0.5).float()
        iou = self.iou(pred_mask, y)
        precision = self.precision(pred_mask, y)
        recall = self.recall(pred_mask, y)
        fscore = self.fscore(pred_mask, y)
        accuracy = self.acc(pred_mask, y)

        metrics = {
            "val_loss": loss,
            "val_iou": iou,
            "val_precision": precision,
            "val_recall": recall,
            "val_fscore": fscore,
            "val_accuracy": accuracy,
        }
        self.log_dict(metrics, logger=True, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)

        if self.validation_log_images_idx % self.tb_log_train_val_images[1]== 0:
            self.log_images(x, y, pred_mask, idx, self.validation_log_images_idx, mode="val")


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Define the test step."""
        
        self.test_log_images_idx += 1
        x, y, idx = batch
        x = x.to(memory_format=torch.channels_last)
        x = x.float()
        y = y.float()

        preds = self.model.forward(x)
        loss = self.loss(preds, y)

        pred_mask = (preds > 0.5).float()
        iou = self.iou(pred_mask, y)
        precision = self.precision(pred_mask, y)
        recall = self.recall(pred_mask, y)
        fscore = self.fscore(pred_mask, y)
        accuracy = self.acc(pred_mask, y)
        metrics = {
            "test_loss": loss,
            "test_iou": iou,
            "test_precision": precision,
            "test_recall": recall,
            "test_fscore": fscore,
            "test_accuracy": accuracy,
        }
        self.log_dict(metrics, logger=True, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log_images(x, y, pred_mask, idx, self.test_log_images_idx, mode="test")



    def configure_optimizers(self, use_lr_scheduler=True):
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if use_lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.t_max) #t_max corresponds to iterations

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                },
            }
        else:
            return optimizer
        
    def add_black_border(self,ax):
        rect = patches.Rectangle((0, 0), 256, 256, linewidth=0.5, edgecolor='black', facecolor='none')
        rect.set_clip_on(False)  # Ensure rectangle is drawn fully
        ax.add_patch(rect)
    def add_black_border_2(self, ax, crop_size):
        rect = patches.Rectangle((0, 0), int(crop_size), int(crop_size), linewidth=0.5, edgecolor='black', facecolor='none')
        rect.set_clip_on(False)  # Ensure rectangle is drawn fully
        ax.add_patch(rect)

    def log_images(self, x, y, pred_mask, sample_ids, idx, mode):
        """Log images to tensorboard."""
        # get a random sample from the batch
        sample = np.random.randint(0, x.shape[0])

        # just log the first band (MWIR) of x
        x = x[sample, 0].cpu()
        x = x[np.newaxis, ...]
        if not self.normalize_mwir_lwir_bands:
            x = revert_feature_standardization(x, self.global_mean, self.global_std)

        y = y[sample, 0].cpu()
        y = y[np.newaxis, ...]

        pred_mask = pred_mask[sample, 0].cpu()
        pred_mask = pred_mask[np.newaxis, ...]

        sample_id = sample_ids[sample]

        plt.rcParams["font.family"] = 'serif'
        plt.rcParams["font.size"] = 11

        fig, axs = plt.subplots(1, 3, figsize=(10,3))
        axs[0].imshow(x[0], cmap = "viridis")
        axs[0].title.set_text(f'Input Data: {sample_id}')

        axs[1].imshow(pred_mask[0], cmap = ListedColormap(["white", "green"]))
        axs[1].title.set_text(f'Model Prediction: {sample_id}')
        self.add_black_border(axs[1])

        axs[1].title.set_text(f'Model Prediction:{sample_id}')
        axs[2].imshow(y[0], cmap = ListedColormap(["white", "green"]))
        axs[2].title.set_text(f'Fire Mask: {sample_id}')
        self.add_black_border(axs[2])

        for ax in axs:
            ax.axis('off')  # Hide axes

        # create a grid of images
        # grid = torchvision.utils.make_grid([x,pred_mask, y], nrow=3, pad_value=1)
        self.logger.experiment.add_figure(f"{mode}_images", fig, idx)
        if mode == "test":
            dir_name = pathlib.Path(f"outputs/test_dataset_plots/{self.exp_name}")
            dir_name.mkdir(parents=True, exist_ok=True)
            # tight layout
            plt.tight_layout()
            fig.savefig(f"{dir_name}/image_{sample_id}.png")

  
# used to log the metrics and loss of the best epoch
class BestMetricCheckpoint(Callback):
    def __init__(self, monitor='val_iou', mode='max'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch_metrics = {}
        self.best_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is not None:
            if ((self.mode == 'min' and current_value < self.best_value)
                    or (self.mode == 'max' and current_value > self.best_value)):
                self.best_value = current_value
                self.best_epoch_metrics = trainer.callback_metrics
                self.best_epoch = trainer.current_epoch

    
    def on_test_end(self, trainer, pl_module):
        # Add the test metrics at the end of testing
        self.test_metrics = trainer.callback_metrics
