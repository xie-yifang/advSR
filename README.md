# Enhancing Stealth and Fidelity of Adversarial Examples via Super-Resolution Reconstruction


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8%2B-orange)](#requirements)  
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-red)](#requirements)

---


This repository contains the implementation for **adv-SR**, a framework that combines super-resolution (ESRGAN-based) with adversarial example generation to produce high-resolution, visually realistic adversarial images while preserving attack effectiveness.


---

## Table of Contents 

- [Usage ](#usage)  
- [License ](#license)  

---

## Features

- Generate high-resolution adversarial examples with improved visual fidelity.  
- Maintain adversarial semantic consistency via a class-constrained loss.  
- Compatible with multiple mainstream attack methods (e.g., FGSM, PGD, BIM, CW, etc.).  
- Evaluation scripts for LPIPS / SSIM / PSNR and attack success rates.  
- Example scripts for training, inference and quality assessment.

---

## Usage

1. Clone repository:

```bash
git clone https://github.com/xie-yifang/advSR.git
cd advSR
```

2. Prepare Data

DIV2K Dataset
Download **DIV2K_train_HR.zip** (800 high-resolution images). 
Extract the dataset and place it under `./data/DIV2K_train_HR/`.

```bash
python scripts/split_images.py \
  --input_dir ./data/DIV2K_train_HR \
  --train_dir ./data/DIV2K_train_GT \
```
Test Set

The Set5 dataset is already included in the ./data/Set5/ folder and can be used directly for testing.

3. Train:

```bash
python train_net.py --config_path ./configs/train/RRDBNet_x2-DIV2K.yaml

python train_gan.py --config_path ./configs/train/ESRGAN_x2-DIV2K.yaml
```
4. Test:

```bash
python test.py --config_path --config_path ./configs/test/ESRGAN_x2-imagenet.yaml

```
```bibtex
@misc{wang2018esrgan,
    title={ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks},
    author={Xintao Wang and Ke Yu and Shixiang Wu and Jinjin Gu and Yihao Liu and Chao Dong and Chen Change Loy and Yu Qiao and Xiaoou Tang},
    year={2018},
    eprint={1809.00219},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


