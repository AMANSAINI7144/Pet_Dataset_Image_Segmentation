"""THIS CODE MISSING THE DOWNLOADING AND EXTRACTING SECTION OF THE DATASET"""
"""python run_unet.py --arch plain"""
"""python run_unet.py --arch attention --epochs 100 --batch-size 8 --run-name my_expt"""

#!/usr/bin/env python3
"""
run_unet.py
Single-entry training script for plain/attention UNet.
Place this in the same folder as dataset.py, transforms.py, losses.py, unet_plain.py, unet_segmentation.py
Usage examples:
  python run_unet.py --arch plain --epochs 100
  python run_unet.py --arch attention --run-name myrun --save-every 5
"""

import os
import time
import csv
import logging
import argparse
import traceback
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# import your modules (must be in PYTHONPATH / same folder)
from dataset import OxfordPetsDataset
from transforms import train_transform, val_transform
from losses import CombinedLoss
from unet_plain import UNet
from unet_segmentation import AttentionUNet

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['plain', 'attention'], default='plain')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--runs-root', type=str, default='runs')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-tensorboard', action='store_true')
    p.add_argument('--device', type=str, default=None, help='cuda or cpu (auto if not set)')
    return p.parse_args()

# ----------------------
# Helpers
# ----------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_run_dirs(runs_root, arch, run_name=None):
    run_id = run_name if run_name else time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(runs_root, arch + '_' + f"run_{run_id}")
    checkpoints = os.path.join(base, "checkpoints")
    results = os.path.join(base, "results")
    logs = os.path.join(base, "logs")
    tb = os.path.join(logs, "tensorboard")
    models = os.path.join(base, "models")
    for d in (checkpoints, results, logs, tb, models):
        os.makedirs(d, exist_ok=True)
    return {
        'BASE': base, 'CHECKPOINTS': checkpoints, 'RESULTS': results,
        'LOGS': logs, 'TENSORBOARD': tb, 'MODELS': models, 'RUN_ID': run_id
    }

def setup_logging(log_path, level=logging.INFO):
    logger = logging.getLogger(f"run_{os.path.basename(log_path)}")
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def build_dataloaders(data_root, batch_size, num_workers):
    train_ds = OxfordPetsDataset(images_dir=os.path.join(data_root, "images"),
                                 masks_dir=os.path.join(data_root, "annotations/trimaps"),
                                 transform=train_transform)
    val_ds = OxfordPetsDataset(images_dir=os.path.join(data_root, "images"),
                               masks_dir=os.path.join(data_root, "annotations/trimaps"),
                               transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(train_ds), len(val_ds)

def build_model(arch, device):
    if arch == 'plain':
        model = UNet(in_channels=3, num_classes=3)
    else:
        model = AttentionUNet(in_channels=3, num_classes=3)
    return model.to(device)

# metrics (same as before)
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

def safe_next_loader_batch(loader):
    try:
        return next(iter(loader))
    except StopIteration:
        return None, None

def save_visual(epoch, model, loader, results_dir, writer, logger, device):
    model.eval()
    imgs, masks = safe_next_loader_batch(loader)
    if imgs is None:
        logger.warning("No validation batch for visualization")
        return
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

    img_np = imgs[0].permute(1, 2, 0).cpu().numpy()
    gt_np = masks[0].cpu().numpy()
    pred_np = preds[0].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    try:
        if img_np.max() <= 1.0:
            ax[0].imshow((img_np * 255).astype('uint8'))
        else:
            ax[0].imshow(img_np.astype('uint8'))
    except Exception:
        ax[0].imshow(img_np)
    ax[0].set_title("Input")
    ax[1].imshow(gt_np)
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_np)
    ax[2].set_title("Prediction")
    for a in ax: a.axis("off")
    plt.tight_layout()

    out_png = os.path.join(results_dir, f"epoch_{epoch}_vis.png")
    try:
        fig.savefig(out_png)
        if writer:
            writer.add_figure("val/Prediction_vs_GT", fig, global_step=epoch)
        logger.info(f"Saved visualization: {out_png}")
    except Exception as e:
        logger.warning(f"Failed to save visualization: {e}")
    finally:
        plt.close(fig)

# ----------------------
# Train loop
# ----------------------
def train_loop(args, model, train_loader, val_loader, paths, logger):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    writer = None if args.no_tensorboard else SummaryWriter(log_dir=paths['TENSORBOARD'])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    metrics_csv = os.path.join(paths['LOGS'], "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_iou", "val_iou", "train_dice", "val_dice"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0; train_iou = 0.0; train_dice = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += iou_score(outputs, masks)
            train_dice += dice_score(outputs, masks)

        # validation
        model.eval()
        val_loss = 0.0; val_iou = 0.0; val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_iou += iou_score(outputs, masks)
                val_dice += dice_score(outputs, masks)

        # average
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        train_iou /= max(1, len(train_loader))
        val_iou /= max(1, len(val_loader))
        train_dice /= max(1, len(train_loader))
        val_dice /= max(1, len(val_loader))

        logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                    f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f} | "
                    f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")

        if writer:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("IoU/Train", train_iou, epoch)
            writer.add_scalar("IoU/Val", val_iou, epoch)
            writer.add_scalar("Dice/Train", train_dice, epoch)
            writer.add_scalar("Dice/Val", val_dice, epoch)

        # append csv
        try:
            with open(metrics_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss, train_iou, val_iou, train_dice, val_dice])
        except Exception as e:
            logger.warning(f"Failed to append CSV: {e}")

        # visual
        save_visual(epoch, model, val_loader, paths['RESULTS'], writer, logger, device)

        # checkpoint
        if epoch % args.save_every == 0:
            ckpt = os.path.join(paths['CHECKPOINTS'], f"epoch_{epoch}.pth")
            try:
                torch.save(model.state_dict(), ckpt)
                logger.info(f"Saved checkpoint: {ckpt}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        # best & early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(paths['MODELS'], "best_model.pth")
            try:
                torch.save(model.state_dict(), best_path)
                logger.info(f"New best model saved: {best_path}")
            except Exception as e:
                logger.warning(f"Failed to save best model: {e}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{args.patience} epochs")
            if epochs_no_improve >= args.patience:
                logger.info("Early stopping triggered")
                break

    if writer:
        writer.close()

# ----------------------
# Main
# ----------------------
def main():
    args = parse_args()
    # device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    set_seed(args.seed)
    arch_folder = 'plain_unet' if args.arch == 'plain' else 'attention_unet'
    paths = make_run_dirs(args.runs_root, arch_folder, run_name=args.run_name)
    logger = setup_logging(os.path.join(paths['LOGS'], "training.log"))
    logger.info(f"RUN ID: {paths['RUN_ID']}")
    logger.info(f"Arch: {args.arch}, Device: {args.device}")
    logger.info(f"Run base: {paths['BASE']}")

    train_loader, val_loader, train_size, val_size = build_dataloaders(args.data_root, args.batch_size, args.num_workers)
    logger.info(f"Train size: {train_size}, Val size: {val_size}")

    model = build_model(args.arch, args.device)
    train_loop(args, model, train_loader, val_loader, paths, logger)
    logger.info("Training finished")

if __name__ == '__main__':
    main()
