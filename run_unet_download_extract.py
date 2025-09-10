#!/usr/bin/env python3
"""
run_unet.py -- Single-file UNet training entrypoint (plain and attention variants)
- Auto-downloads Oxford-IIIT Pet dataset if not present
- Creates run folders: runs/<arch>/run_<YYYYmmdd-HHMMSS>/{checkpoints,results,logs,models}
- Logs to console + runs/.../logs/training.log, metrics.csv
- TensorBoard logging
- Visualizations saved per epoch as PNG
- Default behavior: same as previous scripts; simply run `python run_unet.py`
"""

import os
import sys
import time
import csv
import math
import logging
import argparse
import random
import tarfile
import traceback
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T

# ----------------------------
# Simple UNet and Attention UNet (compact)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_c, out_c))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_c, out_c)
        else:
            self.up = nn.ConvTranspose2d(in_c//2, in_c//2, 2, stride=2)
            self.conv = DoubleConv(in_c, out_c)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding in case sizes mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_c=32):
        super().__init__()
        c = base_c
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.down4 = Down(c*8, c*8)
        self.up1 = Up(c*16, c*4)
        self.up2 = Up(c*8, c*2)
        self.up3 = Up(c*4, c)
        self.up4 = Up(c*2, c)
        self.outc = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# Attention block (simple)
class AttentionBlock(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False),
                                 nn.BatchNorm2d(inter_ch))
        self.W_x = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False),
                                 nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(UNet):
    def __init__(self, in_channels=3, num_classes=3, base_c=32):
        super().__init__(in_channels, num_classes, base_c)
        # Replace some skip connections with attention
        inter = base_c*4
        self.att4 = AttentionBlock(g_ch=base_c*8, x_ch=base_c*8, inter_ch=inter)
        self.att3 = AttentionBlock(g_ch=base_c*4, x_ch=base_c*4, inter_ch=base_c*2)
        self.att2 = AttentionBlock(g_ch=base_c*2, x_ch=base_c*2, inter_ch=base_c)
        # override forward to use attention
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # attention gating before concatenation
        x4_att = self.att4(x5, x4)
        x = self.up1(x5, x4_att)
        x3_att = self.att3(x, x3)
        x = self.up2(x, x3_att)
        x2_att = self.att2(x, x2)
        x = self.up3(x, x2_att)
        x = self.up4(x, x1)
        return self.outc(x)

# ----------------------------
# Dataset: Oxford-IIIT Pets downloader + loader
# ----------------------------
OXFORD_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
OXFORD_ANN_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

def download_and_extract(url, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = url.split('/')[-1]
    out_path = dest_dir / fname
    if not out_path.exists():
        print(f"Downloading {url} -> {out_path} ...")
        urlretrieve(url, out_path)
    else:
        print(f"Found {out_path}, skipping download.")
    # extract
    try:
        with tarfile.open(out_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
    except Exception as e:
        print("Extraction failed:", e)
        raise

class OxfordPetsDataset(Dataset):
    """
    Expects directory structure:
      data/images/*.jpg
      data/annotations/trimaps/*.png
    If missing, the script will try to download and extract the dataset automatically.
    Masks (trimaps) have values 1,2,3 — we map them to 0,1,2 (or as required).
    """
    def __init__(self, root="data", images_dir="images", masks_dir="annotations/trimaps",
                 transform=None, image_size=(512,512), download_if_missing=True):
        self.root = Path(root)
        self.img_dir = self.root / images_dir
        self.mask_dir = self.root / masks_dir
        self.transform = transform
        self.image_size = image_size

        # ensure dataset present
        if download_if_missing:
            if not (self.img_dir.exists() and any(self.img_dir.iterdir())):
                print("Images not found - downloading Oxford-IIIT Pets images...")
                download_and_extract(OXFORD_IMAGES_URL, self.root)
            if not (self.mask_dir.exists() and any(self.mask_dir.iterdir())):
                print("Annotations not found - downloading Oxford-IIIT Pets annotations...")
                download_and_extract(OXFORD_ANN_URL, self.root)

        # build file list by matching filenames
        imgs = sorted([p for p in self.img_dir.glob("*.jpg")])
        masks = sorted([p for p in self.mask_dir.glob("*.png")])
        # match by base name (some masks have suffixes), so map by stem
        mask_map = {m.stem.split('.')[0]: m for m in masks}
        pairs = []
        for img in imgs:
            base = img.stem
            # mask filenames sometimes include class suffix; try to find best match
            if base in mask_map:
                pairs.append((img, mask_map[base]))
            else:
                # try variants
                cand = [m for k,m in mask_map.items() if k.startswith(base)]
                if cand:
                    pairs.append((img, cand[0]))
        self.pairs = pairs
        if len(self.pairs) == 0:
            raise RuntimeError(f"No image-mask pairs found in {self.img_dir} and {self.mask_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, mask_p = self.pairs[idx]
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p)  # trimap values 1,2,3 -> map to 0,1,2
        # ensure mask is single-channel
        mask = mask.convert("L")
        # resize with nearest for mask
        if self.transform is not None:
            img = self.transform['img'](img)
            mask = self.transform['mask'](mask)
        else:
            # default transforms (resize + to tensor)
            img = img.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)
            img = T.ToTensor()(img)
            mask = torch.from_numpy(np.array(mask)).long()
        # remap mask values: original trimap often uses 1,2,3 => subtract 1 to get 0..2
        mask_np = np.array(mask)
        # If mask has values 1..3, convert to 0..2. If values 0..2 already, keep as is.
        if mask_np.max() > 2:
            # sometimes annotation uses 0..255; threshold & remap
            # we'll map pixels: 1->0 (background?), 2->1 (foreground?), 3->2 (edge) depending on dataset conventions.
            mask_np = mask_np // (256 // 3)
        # force in 0..2
        mask_np = (mask_np - mask_np.min()).astype(np.int64)
        mask = torch.from_numpy(mask_np).long()
        # Normalize image and return
        return img, mask

# ----------------------------
# Loss: Combined CE + Dice (simple)
# ----------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, preds, targets, ignore_index=None):
        # preds: raw logits [B,C,H,W]
        preds = nn.functional.softmax(preds, dim=1)
        preds_flat = preds.view(preds.size(0), preds.size(1), -1)
        targets_onehot = nn.functional.one_hot(targets, num_classes=preds.size(1)).permute(0,3,1,2).contiguous()
        targets_flat = targets_onehot.view(targets_onehot.size(0), targets_onehot.size(1), -1).float()
        intersection = (preds_flat * targets_flat).sum(-1)
        denom = preds_flat.sum(-1) + targets_flat.sum(-1)
        dice = (2. * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    def forward(self, preds, targets):
        l_ce = self.ce(preds, targets)
        l_dice = self.dice(preds, targets)
        return self.weight_ce * l_ce + self.weight_dice * l_dice

# ----------------------------
# Metrics
# ----------------------------
def iou_score(preds, targets, num_classes=3):
    preds = torch.argmax(preds, dim=1)
    iou = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        inter = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou.append(1.0)
        else:
            iou.append(inter / union)
    return sum(iou) / len(iou)

def dice_score(preds, targets, num_classes=3):
    preds = torch.argmax(preds, dim=1)
    dice = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        inter = (pred_inds & target_inds).sum().item()
        denom = pred_inds.sum().item() + target_inds.sum().item() + 1e-8
        dice.append((2. * inter) / denom)
    return sum(dice) / len(dice)

# ----------------------------
# Utilities: run folders, logging, visualization
# ----------------------------
def make_run_dirs(root, arch, run_name=None):
    run_id = run_name if run_name else time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(root, arch, f"run_{run_id}")
    checkpoints = os.path.join(base, "checkpoints")
    results = os.path.join(base, "results")
    logs = os.path.join(base, "logs")
    tb = os.path.join(logs, "tensorboard")
    models = os.path.join(base, "models")
    for d in (checkpoints, results, logs, tb, models):
        os.makedirs(d, exist_ok=True)
    return {
        "BASE": base, "CHECKPOINTS": checkpoints, "RESULTS": results,
        "LOGS": logs, "TENSORBOARD": tb, "MODELS": models, "RUN_ID": run_id
    }

def setup_logger(logfile, level=logging.INFO):
    logger = logging.getLogger(f"unet_train_{os.path.basename(logfile)}")
    logger.setLevel(level)
    # Avoid doubling handlers if called twice
    if not logger.handlers:
        fh = logging.FileHandler(logfile)
        fh.setLevel(level)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def save_visual(epoch, model, loader, results_dir, writer, logger, device):
    model.eval()
    try:
        batch = next(iter(loader))
    except Exception:
        logger.warning("Validation loader empty - skipping visualization")
        return
    imgs, masks = batch
    imgs = imgs.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    # convert first example to images
    img0 = imgs[0].cpu().permute(1,2,0).numpy()
    # img0 is tensor in [0,1] likely
    if img0.max() <= 1.0:
        img_disp = (img0 * 255).astype('uint8')
    else:
        img_disp = img0.astype('uint8')
    gt0 = masks[0].cpu().numpy()
    pred0 = preds[0].astype('uint8')
    # plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,3,figsize=(12,4))
        ax[0].imshow(img_disp)
        ax[0].set_title("Input")
        ax[1].imshow(gt0)
        ax[1].set_title("GT")
        ax[2].imshow(pred0)
        ax[2].set_title("Pred")
        for a in ax: a.axis('off')
        out_png = os.path.join(results_dir, f"epoch_{epoch}_vis.png")
        fig.savefig(out_png, bbox_inches='tight')
        if writer:
            writer.add_figure("val/Prediction_vs_GT", fig, global_step=epoch)
        plt.close(fig)
        logger.info(f"Saved visualization: {out_png}")
    except Exception as e:
        logger.warning(f"Failed to save visualization: {e}")

# ----------------------------
# Training loop
# ----------------------------
def train(
    arch="plain",
    data_root="data",
    runs_root="runs",
    image_size=(512,512),
    batch_size=4,
    epochs=200,
    lr=1e-4,
    save_every=10,
    patience=15,
    num_workers=4,
    seed=42,
    run_name=None,
    no_tensorboard=False,
    device=None
):
    # device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # seed
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    arch_folder = "plain_unet" if arch=="plain" else "attention_unet"
    paths = make_run_dirs(runs_root, arch_folder, run_name=run_name)
    logfile = os.path.join(paths["LOGS"], "training.log")
    logger = setup_logger(logfile)
    logger.info(f"RUN ID: {paths['RUN_ID']} - arch={arch} - device={device}")
    logger.info(f"Run base: {paths['BASE']}")

    # transforms: produce image tensors in 0..1 and masks as LongTensor (H,W)
    transform_img = T.Compose([
        T.Resize(image_size, interpolation=Image.BILINEAR),
        T.ToTensor(),  # floats in [0,1]
    ])
    transform_mask = T.Compose([
        T.Resize(image_size, interpolation=Image.NEAREST),
        T.ToTensor(),  # this yields float in [0,1] but for masks we'll convert below
        T.Lambda(lambda x: (x*255).to(torch.long).squeeze(0))  # map back to integer labels
    ])
    transforms = {'img': transform_img, 'mask': transform_mask}

    # dataset
    ds_train = OxfordPetsDataset(root=data_root, transform=transforms, download_if_missing=True)
    ds_val = OxfordPetsDataset(root=data_root, transform=transforms, download_if_missing=False)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info(f"Train size: {len(ds_train)}, Val size: {len(ds_val)}")

    # model
    if arch == "plain":
        model = UNet(in_channels=3, num_classes=3, base_c=32)
    else:
        model = AttentionUNet(in_channels=3, num_classes=3, base_c=32)
    model = model.to(device)
    criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # tensorboard writer
    writer = None if no_tensorboard else SummaryWriter(log_dir=paths["TENSORBOARD"])

    # csv metrics file
    metrics_csv = os.path.join(paths["LOGS"], "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_iou", "val_iou", "train_dice", "val_dice"])

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(paths["MODELS"], "best_model.pth")

    try:
        for epoch in range(1, epochs+1):
            model.train()
            tloss = 0.0; tiou = 0.0; tdice = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
            for imgs, masks in loop:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                tloss += loss.item()
                tiou += iou_score(outputs, masks)
                tdice += dice_score(outputs, masks)

            # validation
            model.eval()
            vloss = 0.0; viou = 0.0; vdice = 0.0
            with torch.no_grad():
                for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                    vloss += loss.item()
                    viou += iou_score(outputs, masks)
                    vdice += dice_score(outputs, masks)

            # average over loaders (guard divide by zero)
            tloss /= max(1, len(train_loader))
            vloss /= max(1, len(val_loader))
            tiou /= max(1, len(train_loader))
            viou /= max(1, len(val_loader))
            tdice /= max(1, len(train_loader))
            vdice /= max(1, len(val_loader))

            logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {tloss:.4f}, Val Loss: {vloss:.4f} | "
                        f"Train IoU: {tiou:.4f}, Val IoU: {viou:.4f} | Train Dice: {tdice:.4f}, Val Dice: {vdice:.4f}")

            # TensorBoard scalars
            if writer:
                writer.add_scalar("Loss/Train", tloss, epoch)
                writer.add_scalar("Loss/Val", vloss, epoch)
                writer.add_scalar("IoU/Train", tiou, epoch)
                writer.add_scalar("IoU/Val", viou, epoch)
                writer.add_scalar("Dice/Train", tdice, epoch)
                writer.add_scalar("Dice/Val", vdice, epoch)

            # append metrics
            try:
                with open(metrics_csv, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, tloss, vloss, tiou, viou, tdice, vdice])
            except Exception as e:
                logger.warning(f"Failed to append metrics CSV: {e}")

            # visualization
            save_visual(epoch, model, val_loader, paths["RESULTS"], writer, logger, device)

            # save checkpoint every save_every
            if epoch % save_every == 0:
                ckpt_path = os.path.join(paths["CHECKPOINTS"], f"epoch_{epoch}.pth")
                try:
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info(f"Saved checkpoint: {ckpt_path}")
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint: {e}")

            # best model & early stopping
            if vloss < best_val_loss:
                best_val_loss = vloss
                epochs_no_improve = 0
                try:
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"New best model saved: {best_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to save best model: {e}")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement for {epochs_no_improve}/{patience} epochs")
                if epochs_no_improve >= patience:
                    logger.info("Early stopping triggered")
                    break

    except KeyboardInterrupt:
        logger.info("Training interrupted by user (KeyboardInterrupt)")
        traceback.print_exc()
    except Exception as e:
        logger.error("Unhandled exception in training loop:")
        logger.error(traceback.format_exc())
    finally:
        if writer:
            writer.close()
        logger.info("Training finished. Outputs saved under: %s", paths["BASE"])

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified UNet training script (single-file).")
    p.add_argument("--arch", choices=["plain", "attention"], default="plain", help="Architecture: plain or attention (default plain)")
    p.add_argument("--data-root", type=str, default="data", help="Dataset root (images and annotations will be under this path)")
    p.add_argument("--runs-root", type=str, default="runs", help="Root directory to store run outputs")
    p.add_argument("--image-size", type=int, default=512, help="Square image size to resize to (default 512)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--device", type=str, default=None, help="Set device (cpu or cuda). If not set, auto-detect.")
    return p.parse_args()

def main():
    args = parse_args()
    image_size = (args.image_size, args.image_size)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train(
        arch=args.arch,
        data_root=args.data_root,
        runs_root=args.runs_root,
        image_size=image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_every=args.save_every,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        run_name=args.run_name,
        no_tensorboard=args.no_tensorboard,
        device=device
    )

if __name__ == "__main__":
    main()
