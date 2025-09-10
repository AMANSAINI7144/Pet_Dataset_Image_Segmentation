import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset



class OxfordPetsDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert mask to numpy
        mask = np.array(mask)
        mask = mask - 1   # from {1,2,3} → {0,1,2}

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure mask is LongTensor for CrossEntropyLoss
        mask = torch.as_tensor(mask, dtype=torch.long)

        return image, mask
