# * 
# * パラメータ管理
# * 

import os
import ipdb

## -_ データセット
TRAIN_DATA_PATH = "./nput/cifar-10-raw/train"
VALIDATION_DATA_PATH = "./input/cifar-10-raw/validation"
## _-

## -_ ハイパーパラメータ
image_resize = 224
num_classes = 10
batch_size = 32
epochs = 30
#*#not_train_layer_n = 19
## _-

## -_ targz解凍直後のデータセットのパス
cifar10_dir = "cifar-10-batches-py/"
## _-

## -_ 訓練後モデルの保存先ディレクトリ
model_save_dir = "result/model"
os.makedirs(model_save_dir, exist_ok=True)
log_dir = "result/log"
os.makedirs(log_dir, exist_ok=True)
png_dir = "result/png"
os.makedirs(png_dir, exist_ok=True)
## _-

