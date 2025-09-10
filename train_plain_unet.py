#!/usr/bin/env python3
import os
import time
import csv
import logging
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from dataset import OxfordPetsDataset
from transforms import train_transform, val_transform
from unet_plain import UNet
from losses import CombinedLoss

# -----------------
# Config
# -----------------
BATCH_SIZE = 4
EPOCHS = 200
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 15  # early stopping patience
SAVE_EVERY = 10  # save checkpoint every N epochs

# -----------------
# Run folder setup (one place to change if you want other naming)
# -----------------
run_id = time.strftime("%Y%m%d-%H%M%S")
RUN_BASE = os.path.join("runs", f"run_{run_id}")             # base run folder
CHECKPOINTS_DIR = os.path.join(RUN_BASE, "checkpoints")     # per-epoch checkpoints
RESULTS_DIR = os.path.join(RUN_BASE, "results")             # png visualizations
LOGS_DIR = os.path.join(RUN_BASE, "logs")                   # logs + metrics + tensorboard
TENSORBOARD_DIR = os.path.join(LOGS_DIR, "tensorboard")
MODELS_DIR = os.path.join(RUN_BASE, "models")               # best / final models

for d in (CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, TENSORBOARD_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

# -----------------
# Logging setup (console + file)
# -----------------
logfile = os.path.join(LOGS_DIR, "training.log")
logger = logging.getLogger("train_logger")
logger.setLevel(logging.INFO)
# Avoid duplicate handlers if reloading
if not logger.handlers:
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# CSV metrics file
metrics_csv = os.path.join(LOGS_DIR, "metrics.csv")
if not os.path.exists(metrics_csv):
    with open(metrics_csv, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "val_loss",
                             "train_iou", "val_iou", "train_dice", "val_dice"])

# TensorBoard writer
writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

logger.info(f"RUN ID: {run_id}")
logger.info(f"Saving run to: {RUN_BASE}")
logger.info(f"Device: {DEVICE}")

# -----------------
# Dataset & Dataloader
# -----------------
train_dataset = OxfordPetsDataset(
    images_dir="data/images",
    masks_dir="data/annotations/trimaps",
    transform=train_transform
)
val_dataset = OxfordPetsDataset(
    images_dir="data/images",
    masks_dir="data/annotations/trimaps",
    transform=val_transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# -----------------
# Model, Loss, Optimizer
# -----------------
model = UNet(in_channels=3, num_classes=3).to(DEVICE)
criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# -----------------
# Metrics functions
# -----------------
def iou_score(preds, targets, num_classes=3):
    preds = torch.argmax(preds, dim=1)
    iou = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou.append(1.0)
        else:
            iou.append(intersection / union)
    return sum(iou) / len(iou)

def dice_score(preds, targets, num_classes=3):
    preds = torch.argmax(preds, dim=1)
    dice = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        denom = pred_inds.sum().item() + target_inds.sum().item() + 1e-8
        dice_coeff = (2 * intersection) / denom
        dice.append(dice_coeff)
    return sum(dice) / len(dice)

# -----------------
# Visualization helper
# -----------------
def save_and_log_images(epoch, model, loader, writer, tag="val"):
    model.eval()
    imgs, masks = next(iter(loader))   # small check-batch
    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
    with torch.no_grad():
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

    # prepare first example
    img_np = imgs[0].permute(1, 2, 0).cpu().numpy()
    gt_np = masks[0].cpu().numpy()
    pred_np = preds[0].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow((img_np * 255).astype('uint8') if img_np.max() <= 1.0 else img_np)
    ax[0].set_title("Input")
    ax[1].imshow(gt_np)
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_np)
    ax[2].set_title("Prediction")
    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR, f"epoch_{epoch}_vis.png")
    fig.savefig(out_png)
    writer.add_figure(f"{tag}/Prediction_vs_GT", fig, global_step=epoch)
    plt.close(fig)
    logger.info(f"Saved visualization: {out_png}")

# -----------------
# Training Loop with Early Stopping
# -----------------
best_val_loss = float("inf")
epochs_no_improve = 0

try:
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0

        # Training
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += iou_score(outputs, masks)
            train_dice += dice_score(outputs, masks)

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        loop_val = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for imgs, masks in loop_val:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_iou += iou_score(outputs, masks)
                val_dice += dice_score(outputs, masks)

        # compute averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_iou /= len(train_loader)
        val_iou /= len(val_loader)
        train_dice /= len(train_loader)
        val_dice /= len(val_loader)

        # logging
        logger.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                    f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f} | "
                    f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")

        # write scalars to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("IoU/Train", train_iou, epoch)
        writer.add_scalar("IoU/Val", val_iou, epoch)
        writer.add_scalar("Dice/Train", train_dice, epoch)
        writer.add_scalar("Dice/Val", val_dice, epoch)

        # append metrics csv
        with open(metrics_csv, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, train_loss, val_loss, train_iou, val_iou, train_dice, val_dice])

        # Visualization every epoch (saves PNG + logs figure to TB)
        save_and_log_images(epoch, model, val_loader, writer, tag="val")

        # save checkpoint every SAVE_EVERY epochs
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

        # early stopping & best model save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(MODELS_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved: {best_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs")
            if epochs_no_improve >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

except KeyboardInterrupt:
    logger.info("Training interrupted by user (KeyboardInterrupt)")
    traceback.print_exc()
except Exception as e:
    logger.error("Unhandled exception during training:")
    logger.error(traceback.format_exc())
finally:
    writer.close()
    logger.info("Training finished, TensorBoard writer closed.")
