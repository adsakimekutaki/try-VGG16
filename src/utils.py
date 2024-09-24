# * 
# * ユーティリティ管理
# * 

import os
import ipdb
import tqdm
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def unpickle(bin_path):
    ''' 生データをロード '''

    with open(bin_path, 'rb') as f:
        raw_data = pickle.load(f, encoding='latin-1')
    return raw_data

def slice_pickle(raw_data, rootdir, label_name_list):
    ''' 生データを解析 '''

    ## -_ 生データの保存先
    for label in label_name_list:
        label_dir   = os.path.join(rootdir, label)
        os.makedirs(label_dir, exist_ok=True)
    ## _-
    
    ## -_ 生データをラベルごとに保存
    pbar            = tqdm.tqdm(range(len(raw_data['filenames'])))
    for i in pbar:
        #ipdb.set_trace()
        filename    = raw_data['filenames'][i]
        label       = label_name_list[raw_data["labels"][i]]
        label_dir   = os.path.join(rootdir, label)
        data        = raw_data['data'][i]
        data        = data.reshape(3, 32, 32)
        data        = np.swapaxes(data, 0, 2)
        data        = np.swapaxes(data, 0, 1)
        with Image.fromarray(data) as img:
            img_out = os.path.join(label_dir, filename)
            img.save(img_out)
            pbar.set_description(img_out)
    ## _-

def export_graph(df, label_postfix='', linestyle='-'):
    ''' Dfをグラフ化 '''

    ### -_ 損失推移
    plt.subplot(121)
    plt.plot(df["loss"], label=f'train{label_postfix}', linestyle=linestyle)
    plt.plot(df["val_loss"], label=f'valid{label_postfix}', linestyle=linestyle)
    plt.title(f"train and valid loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.minorticks_on()
    plt.grid(linewidth=0.5, color='green', linestyle='--', which='both')
    ### _-

    ## -_ 精度推移
    plt.subplot(122)
    plt.plot(df["accuracy"], label=f"train{label_postfix}", linestyle=linestyle)
    plt.plot(df["val_accuracy"], label=f"valid{label_postfix}", linestyle=linestyle)
    plt.title("train and valid accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.minorticks_on()
    plt.ylim(0.0, 1.0)
    plt.grid(linewidth=0.5, color='green', linestyle='--', which='both')
    ## _-

