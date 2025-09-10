# Pet_Dataset_Image_Segmentation
Trained a U-Net model on the Oxford-IIIT Pet Dataset (37 pet categories with pixel-level masks for background, pet, and boundary). Images resized to 512×512, trained for 200 epochs with data augmentation, saving checkpoints and visualizations after every epoch, plus detailed logs and metrics.



# 🐶 U-Net for Oxford-IIIT Pet Dataset Segmentation  

## 📌 Project Description  
This project implements **U-Net based semantic segmentation** on the **Oxford-IIIT Pet Dataset**, which contains 37 pet categories with pixel-level annotations. Each mask has 3 classes:  
- **0**: Background  
- **1**: Pet  
- **2**: Pet Boundary  

All images and masks were resized to **512×512**. Data augmentation (random flips, rotations, color jitter) is applied to training data, while validation/test use resized inputs without augmentation.  

The model was trained for **200 epochs** with:  
- **Loss**: Combined CrossEntropy + Dice Loss  
- **Optimizer**: Adam (lr=1e-4)  
- **Scheduler**: ReduceLROnPlateau (patience=5)  
- **Early stopping**: patience=15  
- **Checkpoints**: saved every epoch, best model saved separately  
- **Visualizations**: input, ground truth, prediction saved after every epoch  

Logs, metrics, and results are saved in structured folders for each run.  

---

## 🚀 Features  
- Plain U-Net (`unet_plain.py`)  
- Training scripts (`train_plain_unet.py`, `train_segmentation_unet.py`)  
- Custom losses (`losses.py`: Dice, Focal, Combined)  
- Data augmentation (`transforms.py`)  
- Automatic logging and visualization  
- Screen session support for long training jobs  

---

## 📂 Repository Structure  
├── dataset.py # Dataset loader
├── losses.py # Loss functions
├── transforms.py # Data augmentation
├── unet_plain.py # U-Net model
├── train_plain_unet.py # Training script for plain U-Net
├── train_segmentation_unet.py# Training script for segmentation U-Net
├── runs/ # Saved runs (checkpoints, results, logs)
└── data/ # Dataset (images + annotations)



---

## ⚡ Training  

Train plain U-Net:  
```bash
python train_plain_unet.py


Train segmentation U-Net:
python train_segmentation_unet.py

Run inside a screen session so it continues after disconnect:

screen -S unet_train
conda activate unet_pet
python train_plain_unet.py


Detach with Ctrl+A, D and reattach with:

screen -r unet_train

🖼️ Results

After every epoch, visualizations are saved in results/ under each run folder.

Logs (training.log) and metrics (metrics.csv) are saved in logs/.

Checkpoints are saved in checkpoints/.

Example run folder structure:

runs/run_20250902_115136/
    ├── checkpoints/
    ├── results/
    ├── logs/

🛠️ Environment

Python 3.10

PyTorch 2.x

NVIDIA RTX A5000 (24GB VRAM)

Install dependencies:

pip install -r requirements.txt

✅ Summary

Dataset: Oxford-IIIT Pet Dataset

Images resized: 512×512

Epochs: 200

Loss: CrossEntropy + Dice

Augmentation: flips, rotations, color jitter

Visualizations + checkpoints after every epoch

Logs and metrics saved per run

