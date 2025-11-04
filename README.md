
# RLCM-DL

## Overview
RLCM-DL is a PyTorch implementation for Refined Lunar Chemistry Mapping, providing deep learning-based solutions for lunar chemical composition analysis. This repository contains the official implementation of the methods described in our upcoming publication.

## System Requirements
The code has been tested on the following environment:
- **Operating System**: Linux (Ubuntu 20.04)

## Installation Guide
```
git clone https://github.com/zyyan-cc/RLCM-DL.git
cd RLCM-DL
pip install -r requirements.txt
```

## Demo
  The user can train the TiO2 inversion model by running the following command:
```
python main_training.py --gpu 0 --dataname TiO2 --batch-size 59 --model-type ResDCNN
```
The user can test the TiO2 inversion model by running the following command:
```
python main_predicting.py --gpu 0 --dataname TiO2 --datapath ./spec_Kaguya_MI_by_row
```

**Important Note**: For testing purposes, please rename your model weights file to `model_best.pth` and place it in the `./model_ckpt/log_TiO2` directory.

### License

`RLCM-DL` is free software made available under the MIT License. For details see the `LICENSE.md` file.
