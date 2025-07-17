# RLCM-DL
## This is a Pytorch implementation for Refined Lunar Chemistry  Mapping (RLCM-DL). The code will be uploaded after the article is published.

### System Requirements
  ###### The code has been tested on Linux (Ubuntu20.04)

### Installation Guide
  ```
https://github.com/zyyan-cc/RLCM-DL.git
cd RLCM-DL
pip install -r requirements.txt
  ```
### Demo
  The user can train the CaO inversion model by running the following command:
```
python main_training.py --gpu 0 --dataname TiO2 --batch-size 59 --model-type ResDCNN
```
### License

`RLCM-DL` is free software made available under the MIT License. For details see the `LICENSE.md` file.
