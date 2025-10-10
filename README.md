# Enhancing Stealth and Fidelity of Adversarial Examples via Super-Resolution Reconstruction

## Abstract
Adversarial examples reveal a critical vulnerability in deep learning systems, yet their practical threat is often undermined by a fundamental trade-off: aggressive perturbation methods that ensure high attack success rates frequently introduce perceptible artifacts, compromising the stealth required to evade detection. This paper addresses a key, underexplored dimension of this problem—the degradation of image resolution inherent in many attack generation pipelines. We introduce adv-SR, a novel framework that decouples adversarial example generation from a subsequent fidelity enhancement phase powered by super-resolution (SR). Our method enhances a baseline ESRGAN model with two key innovations: a class-constrained loss function to preserve the adversarial semantics during reconstruction, and an integrated Squeeze-and-Excitation Network (SENet) module to recalibrate channel-wise features for superior detail recovery. Experiments across five mainstream attack methods demonstrate that adv-SR substantially improves perceptual quality, achieving an average LPIPS reduction of 0.0533, while consistently maintaining or even enhancing attack effectiveness. By transforming low-resolution adversarial examples into high-fidelity counterparts, our work provides a powerful new paradigm for generating stealthy and potent threats, posing significant challenges for next-generation defense systems. The source code for this work is publicly available at https://github.com/xie-yifang/advSR.

## Architecture

<p align="center">
  <a href="images/architecture.png">
    <img src="images/architecture.png" alt="adv-SR Architecture" width="600"/>
  </a>
</p>

---

## Table of Contents 

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Prepare Data](#2-prepare-data)
  - [3. Train](#3-train)
  - [4. Test](#4-test)
  - [5. Evaluation](#5-evaluation)

---

## Features

- Generate high-resolution adversarial examples with improved visual fidelity.  
- Maintain adversarial semantic consistency via a class-constrained loss.  
- Compatible with multiple mainstream attack methods.
- Evaluation scripts for LPIPS / SSIM / PSNR and attack success rates.  
- Example scripts for training, inference and quality assessment.

---

## Requirements
- Python **3.8+**  
- PyTorch **2.1+**  
```bash
pip install -r requirements.txt
```

## Usage

### 1. Clone repository:

```bash
git clone https://github.com/xie-yifang/advSR.git
cd advSR
```

### 2. Prepare Data

DIV2K Dataset
Download **DIV2K_train_HR.zip** (800 high-resolution images)[dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). 
Extract the dataset and place it under `./data/DIV2K_train_HR/`.

```bash
python scripts/split_images.py \
  --input_dir ./data/DIV2K_train_HR \
  --train_dir ./data/DIV2K_train_split \
```
Test Set

- **Set5**:The Set5 dataset is already included in the ./data/Set5/ folder and can be used directly for testing.
- **ImageNet Subset**: From the 1,000 ImageNet classes, 10 images were randomly selected per class to build a smaller ImageNet dataset for adversarial example experiments.  
- **Labels**: The ground-truth class labels are stored in `dev.csv` with the following format:

```csv
ImageId,TrueLabel
0.png,0
1.png,0
2.png,0
3.png,0
4.png,0
```

### 3. Train:

Train RRDBNet:
```bash
python train_net.py --config_path ./configs/train/RRDBNet_x2-DIV2K.yaml
```
Train ESRGAN:
```bash
python train_gan.py --config_path ./configs/train/ESRGAN_x2-DIV2K.yaml
```
### 4. Test:

- **Generating Adversarial Examples**: Use the `attack.py` script to generate adversarial samples. The attacks are implemented using both the [TorchAttacks](https://github.com/Harry24k/adversarial-attacks-pytorch) and [Advertorch](https://github.com/BorealisAI/advertorch) libraries.  
- **Example Attack**: The `attack.py` script demonstrates generating adversarial samples using the **CW** algorithm.  
- **Super-Resolution Reconstruction**: Use the `test.py` script to perform super-resolution on the generated adversarial samples.

Run inference with pre-trained advSR:
```bash
python test.py --config_path ./configs/test/ESRGAN_x2-imagenet.yaml

```

### 5. Evaluation:
Evaluate the quality of reconstructed images using standard metrics:
PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index Measure)
LPIPS (Learned Perceptual Image Patch Similarity)
```bash
python psnr_ssim_lpips.py --original /path/to/original/images --adversarial /path/to/adversarial/images
```
Attack Success Rate for adversarial robustness assessment
```bash
python pred.py --base_folder /path/to/attack/images --csv_path /path/to/attack/dev.csv

```
