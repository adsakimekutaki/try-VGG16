VGG16をお借りし勉強がてら色々試す

*実行環境の立ち上げ
MakefileとDockerfileを読み込み、Dockerコンテナにアタッチ
$ make

*環境設定
入出力や学習エポック数などを指定
@ param.py

*学習実行
$ python train.py
