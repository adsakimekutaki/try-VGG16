FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# NVIDIA container runtime を利用するために必要な設定
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# シンボリックリンクを貼る
RUN ln -s /usr/bin/python3 /usr/bin/python \
  && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# apt
RUN apt update \
  && apt install -y fonts-takao-gothic \
  && apt install -y python3 \
  && apt install -y python3-pip \
  && apt install -y git \
  && apt install -y vim \
  && apt install -y wget \
  && apt install -y unzip \
  && apt install -y libgl1-mesa-dev \
  && apt install -y tree \
  && apt install -y libopencv-dev

# Google Colab から接続できるようにするための extension を追加
RUN python -m pip install --upgrade pip \
  && python -m pip install jupyter_http_over_ws

# 画像処理用のライブラリをインストール
RUN python -m pip install opencv-python \
  && python -m pip install pillow

# tensorflow & kerasをインストール
RUN python -m pip install tensorflow==2.8.0 \
  && python -m pip install keras

# cipy & protobuf をインストール
RUN python -m pip install scipy \
  && python -m pip install protobuf==3.20.1

# ユーティリティをインストール
RUN python -m pip install ipdb \
  && python -m pip install tqdm \
  && python -m pip install pandas \
  && python -m pip install matplotlib

# プロンプトの表示を変更
RUN { \
    echo 'export PS1="${debian_chroot:+($debian_chroot)}\u@\[\e[1;33m\]\w\[\e[m\] $"'; \
    echo ''; \
} >> $HOME_DIR/.bashrc

# ホームディレクトリを追加
ARG HOME_DIR='/work/finetuning'
WORKDIR $HOME_DIR
