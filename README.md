# Res1DNN
## This is a Pytorch implementation for lunar surface oxide inversion. The code will be uploaded after the article is published.

### System Requirements
  ###### The code has been tested on Linux (Ubuntu20.04)

### Installation Guide
  ```
git clone https://github.com/zyyan-cc/Res1DNN
cd Res1DNN
pip install -r requirements.txt
  ```
### Demo
  The user can train the CaO inversion model by running the following command:
```
python main_training.py --gpu 0 --dataname TiO2 --batch-size 59 --model-type ResDCNN
```
### License

`Res1DNN` is free software made available under the MIT License. For details see the `LICENSE.md` file.
