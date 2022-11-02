'''
DNN学習スクリプト
'''

from asyncio import base_tasks
from PIL import Image
import numpy as np
import glob
import os
import cv2 as cv
import time
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import optimizers
from dataloader import dataloader, set_dir
from model.CRNN import crnn
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy

def train(data_root, eval_dataset_id,batch_size, num_epochs, model, loss, optimizer, save_folder, model_name):

    start = time.time()

    # define model
    model.summary()

    # data loader
    print('---------load STFT---------')
    x_train_1, y_train_1, z = dataloader(set_dir(eval_dataset_id, dataset_dir=data_root, train=True))
    x_vali_1, y_vali_1, z1  = dataloader(set_dir(eval_dataset_id, dataset_dir=data_root, train=False))
    
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    json_string = model.to_json()
    open(f'{save_folder}/{model_name}.json', 'w').write(json_string)

    check_point = f'{save_folder}/{model_name}.hdf5'
    callbacks = [ModelCheckpoint(filepath = check_point, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = "auto")]
    
    model.fit(x_train_1, y_train_1, validation_data = (x_vali_1, y_vali_1), epochs = num_epochs, batch_size = batch_size, callbacks = callbacks)
    
    model.save_weights(f'{save_folder}/{model_name}.hdf5')

    process_time = (time.time() - start) / 60
    print(u'学習終了。かかった時間は', process_time, u'分です。')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256, help='set batch size')
    parser.add_argument('--num_epochs', type=int, default=500, help='set num_epochs')
    parser.add_argument('--feats_data_path', type=str, default='./data/feats', help='set save feats data path')
    parser.add_argument('--eval_dataset_id', type=int, help='set eval dataset_id')
    parser.add_argument('--model_name', type=str, default='model', help='set model name')
    args = parser.parse_args()

    batch_size      = args.batch_size
    num_epochs      = args.num_epochs
    data_root       = args.feats_data_path
    eval_dataset_id = args.eval_dataset_id
    model_name      = args.model_name
    
    save_folder = f'./assets/dataset_{eval_dataset_id}/model'
    os.makedirs(save_folder, exist_ok = True)
    
    model     = crnn()
    loss      = weighted_categorical_crossentropy(np.array([0.2, 0.5, 1, 1]))
    optimizer = optimizers.Adam(lr=5e-3, decay=0.005, amsgrad=True)

    train(data_root       = data_root,
          eval_dataset_id = eval_dataset_id,
          batch_size      = batch_size,
          num_epochs      = num_epochs,
          model           = model,
          loss            = loss,
          optimizer       = optimizer,
          save_folder     = save_folder,
          model_name      = model_name)



