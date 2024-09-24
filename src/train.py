import ipdb
import argparse

from model_operation import ModelOperation
from param import *


if __name__ == '__main__':

    #ipdb.set_trace()
    for not_train_layer_n in [15, 19]:

        ## -_ モデルを取得
        main                    = ModelOperation(
            TRAIN_DATA_PATH             = TRAIN_DATA_PATH,
            VALIDATION_DATA_PATH        = VALIDATION_DATA_PATH,
            image_resize                = image_resize,
            num_classes                 = num_classes,
            batch_size                  = batch_size,
            epochs                      = epochs,
            model_save_dir              = model_save_dir,
            log_dir                     = log_dir,
            png_dir                     = png_dir,
            not_train_layer_n           = not_train_layer_n
        )
        train_generator         = main.get_train_data()
        validation_generator    = main.get_valid_data()
        model                   = main.get_vgg16()
        print(' -- VGG16 --')
        model.summary()
        #main.get_trainable(model)
        print(' -- _____ --')
        model                   = main.add_layer(model)
        ## _-

        ## -_ ファインチューニング
        if 19 == not_train_layer_n:
            finetuning          = False
        else:
            finetuning          = True
        model                   = main.set_trainable(model, not_train_layer_n=not_train_layer_n)
        
        msg                     = f'finetuning is :{finetuning}'
        print(msg)
        ## _-

        ## -_ 学習
        print(' -- VGG16 plus --')
        model.summary()
        main.get_trainable(model)
        print(' -- __________ --')
        print('exit to check summary')
        #exit()  ## FIXME 
        main.train(model, train_generator, validation_generator)
        ## _-
