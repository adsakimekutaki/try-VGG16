# *
# * モデルを構築
# * <参考> 
# * [CIFAR10][VGG16] https://qiita.com/ps010/items/dee9413d3de28de7d2f9

from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras.models import Model
import os
import ipdb
import pandas as pd

from param import *
from utils import *


class ModelOperation():
    ''' モデル '''

    def __init__(self, 
        TRAIN_DATA_PATH,
        VALIDATION_DATA_PATH,
        image_resize,
        num_classes,
        batch_size,
        epochs,
        model_save_dir,
        log_dir,
        png_dir,
        not_train_layer_n
    ):
        ''' 初期化 '''

        ## -_ パラメータを取得
        self.TRAIN_DATA_PATH             = TRAIN_DATA_PATH
        self.VALIDATION_DATA_PATH        = VALIDATION_DATA_PATH
        self.image_resize                = image_resize
        self.num_classes                 = num_classes
        self.batch_size                  = batch_size
        self.epochs                      = epochs
        self.model_save_dir              = model_save_dir
        self.log_dir                     = log_dir
        self.png_dir                     = png_dir
        self.not_train_layer_n           = not_train_layer_n
        os.makedirs(self.model_save_dir, exist_ok=True)
        ## _-

    def get_train_data(self):
        ''' 学習用データを取得 '''

        ## -_ 学習用読み込みとデータ拡張
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_generator = train_datagen.flow_from_directory(
                TRAIN_DATA_PATH,
                target_size=(self.image_resize, self.image_resize),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
        ## _-

        return train_generator

    def get_valid_data(self):
        ''' 検証用データを取得 '''

        ## -_ 検証用データの読み込みとデータ拡張
        validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        validation_generator = validation_datagen.flow_from_directory(
                VALIDATION_DATA_PATH,
                target_size=(image_resize, image_resize),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )
        ## _-

        return validation_generator

    def get_vgg16(self):
        ''' VGG16を取得 '''

        ## -_ VGG16のモデルと重みをインポート
        input_tensor = Input(shape=(image_resize, image_resize, 3))
        vgg16_model = VGG16(
            include_top=False, #全結合層を除外
            weights='imagenet', 
            input_tensor=input_tensor
        )

        vgg16_model.summary()

        return vgg16_model

    def add_layer(self, model):
        ''' モデルの全結合層を拡張 '''

        ## -_ 重みを固定
        for layer in model.layers:
            layer.trainable = False
        ## _-

        ## -_ 全結合層を構築
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        ## _-

        ## -_ 元モデルと構築した全結合層を結合
        model = Model(inputs=model.input, outputs=predictions)

        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        ## _-

        return model

    def get_trainable(self, model):
        ''' 学習の可否を表示 '''

        ## -_ 学習の可否を取得
        for layer in model.layers: 
            print(layer.name, layer.trainable)
        ## _-

    def set_trainable(self, model, not_train_layer_n=19):
        ''' 学習の可否を設定 '''

        ## -_ 学習可否の設定
        self.not_train_layer_n  = not_train_layer_n
        for layer in model.layers[not_train_layer_n:]:
            layer.trainable     = True
        for layer in model.layers[:not_train_layer_n]:
            layer.trainable     = False
        ## _-

        return model

    def train(self, model, train_generator, validation_generator):
        ''' 学習 '''

        ## -_ 学習パラメータ
        train_name  = os.path.basename(TRAIN_DATA_PATH)
        train_param = f'{train_name}_vgg16_fc_epochs={self.epochs}_batchSize={self.batch_size}_notTrainLayerN={self.not_train_layer_n}'
        ## _-

        ## -_ 学習を実行
        csvlogger_cb    = CSVLogger(os.path.join(self.log_dir, f'csvLogger_{train_param}.csv'))
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=self.epochs, 
            batch_size=self.batch_size,
            callbacks=[csvlogger_cb]
        )
        ## _-

        ## -_ モデルを保存
        model_save_file = f'{train_param}.h5'
        model_save_path = os.path.join(self.model_save_dir, model_save_file)
        model.save_weights(model_save_path)
        print(f'[SAVE MODEL]{model_save_file}')
        ## _-

        ## -_ 学習ログを保存
        csv_out         = os.path.join(self.log_dir, f'history_{train_param}.csv')
        history_df      = pd.DataFrame(history.history)
        history_df.to_csv(csv_out)
        print(f'[SAVE LOG]{csv_out}')
        ## _- 

        ## -_ 学習曲線を保存
        png_out         = os.path.join(self.png_dir, f'graph_{train_param}.png')
        export_graph(history_df, png_out)
        plt.savefig(png_out)
        print(f'[SAVE PNG]{png_out}')
        ## _-
