VGG16をお借りし勉強がてら色々試す

* 実行環境を起動
  - MakefileとDockerfileを読み込み、Dockerコンテナにアタッチ
    - $ make

* パラメータを設定
  - 入出力や学習エポック数などを指定@param.py

* 学習を実行
  - VGG16をロードし、出力層を書き換え、学習
    - $ python train.py
